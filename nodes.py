"""MiniMax H3 Timeline Editor + combined keyframe/reference conditioning.

Two verified native-ComfyUI bugs prevent keyframe conditioning (fl2va-style
first/last frame) and reference conditioning (ref2va-style @-tagged media)
from being used in the same generation. Both were independently confirmed
against the currently-installed source in this project, not inherited from
any prior analysis:

  1. comfy/model_base.py's MiniMaxH3.extra_conds builds cond_video_latents
     from keyframes, then unconditionally overwrites it with the refs list
     instead of concatenating -- with both present, keyframe latents never
     reach the model.
  2. comfy/ldm/minimax/model.py's PackedLayout anchors a "first" keyframe's
     rotary time to the raw text length, correct only when nothing precedes
     the video segment. Each reference pushes the video's real start later,
     so with references present the keyframe ends up anchored before the
     video actually begins.

This module fixes both by patching the loaded model's extra_conds (the
standard `model.clone().add_object_patch(...)` mechanism, not a core-file
edit) with a corrected version: concatenates both latent lists, and anchors
keyframes against the real post-reference video origin.

REMOVED, deliberately, after extensive same-seed A/B testing: a whole
per-reference anchoring/"channel" system (anchor_seconds/anchor_channel,
anchor_closeness, anchor_decouple_scale_seconds, spatial_collision_offset)
used to exist here, built on the hypothesis that references needed to be
manually pulled toward a shared position to co-occur in one scene. Testing
disproved the premise: plain unanchored multi-reference conditioning (with
just the two bugs above fixed) produced clean co-presence on its own, no
different in quality from any anchored configuration tested. The anchoring
system's real, confirmed effects -- same-position binding vs. drift-apart
segregation, a hard bleed artifact near the clip's tail -- were all real,
just never actually necessary to solve the problem it was built for. Kept
out rather than left in as an unused/misleading knob. See git history for
the removed implementation if a real need for deliberate multi-scene
staging (references drifting into separate moments) comes up later.

Mid-clip keyframes (role keyframe_mid) are unaffected by that removal --
a keyframe's own placement is a real, load-bearing seconds value (it's
literally the frame it renders as), not the reference-only experiment.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Mapping

import folder_paths
import torch
import torchaudio

import node_helpers
import nodes
import comfy.conds
import comfy.ldm.minimax.model as h3model
import comfy.model_base
from comfy_api.latest import InputImpl
from comfy_extras import nodes_minimax_h3 as h3
from comfy_extras.nodes_audio import load as _load_audio_waveform

# --- canvas / reference sizing -----------------------------------------

REF_IMAGE_1K = "1k"
REF_IMAGE_15K = "1.5k"
REF_IMAGE_2K = "2k"
REF_IMAGE_MATCH = "match"
REF_IMAGE_ORIGINAL = "original"
REFERENCE_IMAGE_AREAS = {REF_IMAGE_1K: 1024 * 1024, REF_IMAGE_15K: 1536 * 1536, REF_IMAGE_2K: 2048 * 2048}
REFERENCE_SIZE_SEARCH_RADIUS = 16

RESOLUTION_480 = "480P"
RESOLUTION_CUSTOM = "custom"
RESOLUTION_MEGAPIXELS = {
    "360P": 0.2, "416P": 0.3, RESOLUTION_480: 0.4, "540P": 0.5, "640P": 0.7,
    "720P": 0.9, "768P": 1.0, "832P": 1.2, "928P": 1.5, "1024P": 1.8, "1080P": 2.0,
}
ASPECT_WIDESCREEN = "16:9"
ASPECT_RATIOS = {
    "1:1": (1, 1), "2:3": (2, 3), "3:2": (3, 2), "3:4": (3, 4), "4:3": (4, 3),
    "9:16": (9, 16), ASPECT_WIDESCREEN: (16, 9), "21:9": (21, 9),
}

REFERENCE_PLACEHOLDER_RE = re.compile(r"__MINIMAX_H3_REF_(\d+)__")


def _reference_aligned_size(image_w: int, image_h: int, scale: float) -> tuple[int, int]:
    """Nearest H3-grid-aligned (multiple-of-CANVAS_MULTIPLE) size to image_w/h*scale
    that keeps the aspect ratio close, searched over a small window of candidate
    grid-unit counts rather than solved in closed form (aspect + grid-alignment
    both constrain the result, and naive rounding of one can miss a much better
    joint fit that's only 1-2 grid units away)."""
    multiple = h3.CANVAS_MULTIPLE
    scaled_w = max(float(multiple), image_w * scale)
    scaled_h = max(float(multiple), image_h * scale)
    target_area = scaled_w * scaled_h
    aspect = image_w / max(1, image_h)
    center_h_units = max(1, round(scaled_h / multiple))
    best = None
    for h_units in range(max(1, center_h_units - REFERENCE_SIZE_SEARCH_RADIUS), center_h_units + REFERENCE_SIZE_SEARCH_RADIUS + 1):
        ideal_w_units = h_units * aspect
        min_w_units = max(1, math.floor(ideal_w_units) - 2)
        max_w_units = max(min_w_units, math.ceil(ideal_w_units) + 2)
        for w_units in range(min_w_units, max_w_units + 1):
            target_w, target_h = w_units * multiple, h_units * multiple
            ratio_error = abs((target_w / target_h) / aspect - 1.0)
            area_error = abs((target_w * target_h) / target_area - 1.0)
            candidate = (ratio_error * 20.0 + area_error, ratio_error, area_error, target_w, target_h)
            if best is None or candidate < best:
                best = candidate
    return best[3], best[4]


def _original_reference_size(image_w: int, image_h: int) -> tuple[int, int]:
    """Keep a reference at its original size, only cropping down to the nearest
    H3 grid alignment -- or, if it's smaller than one grid cell, scale it up to
    the smallest usable size instead of rejecting it."""
    multiple = h3.CANVAS_MULTIPLE
    target_w, target_h = (image_w // multiple) * multiple, (image_h // multiple) * multiple
    if target_w >= multiple and target_h >= multiple:
        return target_w, target_h
    scale = max(multiple / max(1, image_w), multiple / max(1, image_h))
    return _reference_aligned_size(image_w, image_h, scale)


def _align_canvas_dimension(value: float) -> int:
    return max(h3.CANVAS_MULTIPLE, round(float(value) / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)


def _canvas_dimensions(resolution: str, aspect_ratio: str, custom_width: int, custom_height: int) -> tuple[int, int]:
    if str(resolution) == RESOLUTION_CUSTOM:
        return _align_canvas_dimension(custom_width), _align_canvas_dimension(custom_height)
    megapixels = RESOLUTION_MEGAPIXELS.get(str(resolution), RESOLUTION_MEGAPIXELS[RESOLUTION_480])
    ratio_w, ratio_h = ASPECT_RATIOS.get(str(aspect_ratio), ASPECT_RATIOS[ASPECT_WIDESCREEN])
    scale = math.sqrt(megapixels * 1024 * 1024 / (ratio_w * ratio_h))
    return _align_canvas_dimension(ratio_w * scale), _align_canvas_dimension(ratio_h * scale)


def _frame_length(seconds: float, fps: float) -> int:
    """Round a requested duration to the nearest valid frame count for H3's
    17-frame/5-token temporal compression cycle (always 5 + a multiple of 17)."""
    target_frames = max(5.0, float(seconds) * float(fps))
    block_count = max(0, round((target_frames - 5) / 17))
    return block_count * 17 + 5


# --- reference video/audio decoding --------------------------------------

def _video_parts(value) -> tuple[torch.Tensor, dict | None, float]:
    """Pull (frames, audio, fps) out of a VIDEO-like value -- a real ComfyUI
    VIDEO object (has get_components), a dict with images/frames, or a raw
    4D image-batch tensor treated as already-decoded frames."""
    if hasattr(value, "get_components"):
        components = value.get_components()
        return components.images, components.audio, float(components.frame_rate or 24.0)
    if isinstance(value, Mapping):
        frames = value.get("images") or value.get("frames")
        if isinstance(frames, torch.Tensor):
            return frames, value.get("audio"), float(value.get("fps") or value.get("frame_rate") or 24.0)
    if isinstance(value, torch.Tensor) and value.ndim == 4:
        return value, None, 24.0
    raise ValueError("Unsupported reference video payload")


def _resample_video_frames(frames: torch.Tensor, source_fps: float) -> torch.Tensor:
    if not source_fps or abs(source_fps - h3.FPS) < 0.01:
        return frames
    count = max(1, round(frames.shape[0] * h3.FPS / source_fps))
    indexes = torch.linspace(0, frames.shape[0] - 1, count, device=frames.device).round().long()
    return frames[indexes]


def _audio_sample_rate(audio: Mapping) -> int:
    return int(audio.get("sample_rate") or audio.get("samplerate") or 32000)


def _encode_reference_audio(audio_vae, audio: Mapping):
    waveform = audio["waveform"]
    sample_rate = _audio_sample_rate(audio)
    vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    return latent, latent.shape[-1]


def _resolve_reference_prompt(prompt: str, tag_by_input: dict[int, str], soundtrack_pairs: list[tuple[int, int]],
                               video_count: int, standalone_audio_count: int) -> str:
    resolved = REFERENCE_PLACEHOLDER_RE.sub(lambda m: tag_by_input.get(int(m.group(1)), ""), str(prompt or ""))
    if soundtrack_pairs and (video_count > 1 or standalone_audio_count > 0):
        provenance = [f"<Audio {a}> is the synchronized audio track of <Video {v}>." for a, v in soundtrack_pairs]
        return "\n".join((*provenance, resolved))
    return resolved


KEYFRAME_START = "keyframe_start"
KEYFRAME_END = "keyframe_end"
KEYFRAME_MID = "keyframe_mid"
REFERENCE = "reference"
ROLES = (KEYFRAME_START, KEYFRAME_END, KEYFRAME_MID, REFERENCE)
KEYFRAME_ROLES = (KEYFRAME_START, KEYFRAME_END, KEYFRAME_MID)
MAX_MEDIA = 40  # matches the old Hybrid project's card UI cap -- not an architectural limit

ANCHOR_UNSET = -1.0  # anchor_seconds sentinel: keyframe_mid uses this to mean "not placed"


def _load_media_file(filename: str, media_type: str):
    """Load a file the frontend already uploaded via /upload/image (same
    endpoint native LoadImage uses) into the tensor/object shape the rest of
    this module's conditioning code expects -- the same job a real
    LoadImage/LoadVideo/LoadAudio graph node normally does, done here
    directly since this node takes uploaded files instead of graph links."""
    if media_type == "image":
        image, _mask = nodes.LoadImage().load_image(filename)
        return image
    if media_type == "video":
        return InputImpl.VideoFromFile(folder_paths.get_annotated_filepath(filename))
    if media_type == "audio":
        waveform, sample_rate = _load_audio_waveform(folder_paths.get_annotated_filepath(filename))
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    raise ValueError(f"Unknown media type: {media_type}")


@dataclass(frozen=True)
class _TimelineItem:
    media_type: str
    filename: str
    role: str
    item_index: int
    anchor_seconds: float = ANCHOR_UNSET  # keyframe_mid placement only; references no longer use this
    # PER-ITEM noise_aug (unlike MiniMaxH3TimelineBundle's global
    # visual_cond_noise_aug/audio_cond_noise_aug) -- see _make_fixed_cond_rows
    # for how this becomes real per-row control, not just a global scalar.
    noise_aug: float = 0.999


@dataclass(frozen=True)
class MiniMaxH3TimelineBundle:
    items: tuple[_TimelineItem, ...]
    duration_seconds: float
    # None of the fields below are exposed as widgets anymore. pretimeline_gap
    # was tested directly (0.3s vs 3.0s, same seed/prompt/references
    # otherwise) and produced no meaningfully different result either --
    # same conclusion as the anchoring system it's a close cousin of. Left as
    # fixed defaults rather than deleted: the packed-layout math that reads
    # these is still fully intact and
    # correct, just never triggered since references no longer set
    # anchor_frame_index. Recoverable by re-exposing widgets, not by
    # rewriting math.
    pretimeline_gap_seconds: float = 1.0
    spatial_collision_offset: float = 64.0
    anchor_decouple_scale_seconds: float = 2.0
    # Native MiniMax H3 parameters, real and exposed (see INPUT_TYPES) --
    # unlike the fields above, these are a single GLOBAL scalar applied
    # uniformly to every keyframe/reference row at once (comfy/ldm/minimax/
    # model.py's _cond_video_rows/_cond_audio_rows read one payload value for
    # the whole generation, not per-row), so there's no meaningful per-item
    # version to expose -- this genuinely is a global setting, not a
    # simplification like pretimeline_gap was. Controls both how much noise
    # gets mixed into each conditioning row (r = aug*r + (1-aug)*noise) and
    # what "denoising timestep" that row appears to be at to the network
    # (max(current_timestep, aug)) -- at the native default 0.999, a keyframe
    # is treated as essentially already-resolved content from the very first
    # sampling step, which is why a mid-clip keyframe can produce a hard cut
    # instead of a gradual transition into it: the model isn't given room to
    # build up to it. Lowering this loosens that at the cost of the keyframe/
    # reference being reproduced less exactly.
    visual_cond_noise_aug: float = 0.999
    audio_cond_noise_aug: float = 1.0


class MiniMaxH3TimelineEditor:
    """Attach images/video/audio and mark each item's timeline role.

    Media is uploaded directly (the same /upload/image endpoint native
    LoadImage uses), not wired in from separate loader nodes -- the same
    approach ComfyUI-MiniMaxH3-Hybrid's timeline UI used, kept here because
    a real graph socket per media slot means LiteGraph renders one input row
    per slot regardless of whether it's connected, which for a 40-slot node
    is a wall of dots no matter how the widgets below it are handled. One
    hidden JSON widget (media_json) holds the real per-item state (filename,
    media_type, role, anchor_seconds, anchor_closeness); the frontend
    (web/minimax_h3_timeline_ui.js) renders and edits it as upload cards and
    this node's job is purely to parse it into a MiniMaxH3TimelineBundle."""

    CATEGORY = "MiniMax H3 Easy/Timeline"
    FUNCTION = "build_timeline"
    RETURN_TYPES = ("MINIMAX_H3_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    DESCRIPTION = "Upload media and mark each item as a keyframe (start/end/mid) or a reference. Wire into MiniMax H3 Conditioning (Timeline Integration)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 15.0, "step": 0.1}),
                # How rigidly EVERY keyframe/reference row is enforced, all at
                # once (native MiniMax H3 parameter, applied globally -- see
                # MiniMaxH3TimelineBundle's docstring comment for the full
                # mechanism). 0.999 is the native default: rows are treated as
                # already-resolved from the first sampling step, which can
                # cause a hard cut into a mid-clip keyframe instead of a
                # gradual transition. Lower to loosen that, at the cost of
                # less exact reproduction of keyframe/reference content.
                "visual_cond_noise_aug": ("FLOAT", {"default": 0.999, "min": 0.0, "max": 1.0, "step": 0.001}),
                # Same mechanism, for reference AUDIO rows specifically.
                "audio_cond_noise_aug": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                # {filename, type, role, anchor_seconds}[] -- populated by the
                # frontend's upload cards, not meant to be hand-edited. See
                # MAX_MEDIA for the (non-architectural) item cap.
                "media_json": ("STRING", {"default": "[]", "multiline": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def build_timeline(self, duration_seconds, visual_cond_noise_aug=0.999, audio_cond_noise_aug=1.0, media_json="[]"):
        try:
            raw_items = json.loads(media_json or "[]")
        except (TypeError, ValueError):
            raw_items = []
        if not isinstance(raw_items, list):
            raw_items = []

        items: list[_TimelineItem] = []
        for index, raw in enumerate(raw_items[:MAX_MEDIA]):
            if not isinstance(raw, dict):
                continue
            filename = str(raw.get("filename") or "").strip()
            media_type = str(raw.get("type") or "").strip().lower()
            if not filename or media_type not in {"image", "video", "audio"}:
                continue
            role = str(raw.get("role") or REFERENCE)
            if role not in ROLES:
                raise ValueError(f"Unknown timeline role: {role}")
            if role == REFERENCE:
                # References no longer anchor at all -- testing showed plain
                # unanchored multi-reference conditioning already produces
                # clean co-presence once the two bugs above are fixed, so
                # there's nothing useful for a reference to set here anymore.
                anchor_seconds = ANCHOR_UNSET
            else:
                # keyframe_mid genuinely needs a real second-level placement
                # (that's the frame it's rendered at) -- unaffected.
                anchor_seconds = float(raw.get("anchor_seconds", ANCHOR_UNSET))
                if role == KEYFRAME_MID and anchor_seconds < 0.0:
                    raise ValueError(f"Timeline item {index} ({filename}) is role keyframe_mid but has no anchor_seconds set")
            # PER-ITEM noise_aug default mirrors the native per-modality
            # default (0.999 visual, 1.0 audio) when the item doesn't set
            # its own value.
            default_noise_aug = 1.0 if media_type == "audio" else 0.999
            item_noise_aug = float(raw.get("noise_aug", default_noise_aug))
            items.append(_TimelineItem(media_type, filename, role, index, anchor_seconds, item_noise_aug))
        keyframe_starts = [i for i in items if i.role == KEYFRAME_START]
        keyframe_ends = [i for i in items if i.role == KEYFRAME_END]
        if len(keyframe_starts) > 1 or len(keyframe_ends) > 1:
            raise ValueError("MiniMax H3 Timeline Editor accepts at most one keyframe_start and one keyframe_end item")
        return (MiniMaxH3TimelineBundle(
            tuple(items), float(duration_seconds),
            visual_cond_noise_aug=float(visual_cond_noise_aug),
            audio_cond_noise_aug=float(audio_cond_noise_aug),
        ),)


def _frame_index_to_token_index(pixel_frame_index):
    """Map a pixel frame index to the latent-token index whose block contains
    it. MiniMax H3's temporal VAE compresses frames in a repeating 5-token,
    17-frame cycle (comfy.ldm.minimax.model.FRAME_PER_TOKEN = (1, 4, 4, 4,
    4)): token 0 of each cycle covers exactly 1 pixel frame, tokens 1-4 each
    cover 4 pixel frames compressed together. A request lands exactly on its
    own rotary time slot only at cycle boundaries (frame 0, 17, 34, ...);
    anything inside a 4-frame block resolves to that block's shared slot
    (the block's start) -- there's no finer time resolution to snap to."""
    cycle, r = divmod(pixel_frame_index, 17)
    subtoken = 0 if r == 0 else 1 + (r - 1) // 4
    return cycle * 5 + subtoken


def _corrected_packed_layout(text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None,
                              frame_count=None, pretimeline_gap=None, spatial_collision_offset=64.0,
                              anchor_decouple_scale=None):
    """Same construction as comfy.ldm.minimax.model.PackedLayout, reusing its
    own helper functions so this stays in lockstep with core's math -- with
    three changes, all real experiments with no verified-correct behavior to
    compare against, since neither checkpoint was trained on this
    combination and there's no reference implementation:

      1. THE FIX for the keyframe/reference time-origin bug: a keyframe's
         rotary anchor is computed against the REAL video origin (text_len
         plus however far the reference loop's cursor advanced, plus
         pretimeline_gap), not the raw text length.
      2. Mid-clip keyframes: resolved_frame_index is no longer restricted to
         0/frame_count-1 -- any pixel index maps to its real
         compression-token time (see _frame_index_to_token_index).
      3. Per-reference anchoring: an image/audio/video reference block may
         carry "anchor_frame_index" (+ "anchor_closeness") to pull its
         rotary time toward a specific point in the target video's timeline
         instead of the generic sequential pre-timeline slot every reference
         otherwise gets -- this is the mechanism for making two references
         genuinely co-occur with the same moment instead of just each
         independently "existing somewhere" in the clip. Two anchored refs
         resolving to the exact same time get separated on the spatial
         (h, w) axes by spatial_collision_offset instead of colliding.

         IMPORTANT, per direct testing: "anchor" here is architecturally a
         RoPE relative-distance ATTENTION BIAS, not a hard scheduling gate --
         generation is one joint non-sequential denoise over the whole clip,
         not frame by frame, so there's no mechanism that could make an
         identity appear ONLY at/after its anchor time, and in practice a
         reference anchored at 2s was observed already present (entering
         frame) well before that, around 0.8s. That said, don't undersell it
         either: the same test showed real, visible build-up in that
         identity's presence/prominence culminating near the anchor time --
         a soft attention bias can still produce a genuine visible timing
         effect, it's just not a precise "appears at exactly N seconds"
         placement. Net: treat anchor_seconds as "roughly biases where in the
         clip this reference is strongest," not as a precise appearance-time
         control and not as having no visible timing effect either -- both
         overclaims are wrong. The precise shape of that bias (how early
         presence starts ramping up, how sharply it concentrates) is not
         yet characterized; a same-seed anchor-near-start vs anchor-near-end
         comparison would pin it down further.

    CLOSENESS BUG, found via that comparison and fixed here: closeness used
    to be a FRACTION of the distance from the reference's unanchored position
    to its anchor target (final_t = unanchored_t + closeness * (exact_t -
    unanchored_t)). That distance grows with how far into the clip
    anchor_seconds points, so the exact same closeness value pulled harder
    (stronger real-video locality, more background-bleed risk) the later
    anchor_seconds was set to -- confirmed with a same-seed, closeness-held-
    constant, anchor-only-changed comparison (anchor=4s produced visibly
    stronger reference-content transfer AND a background/identity bleed
    artifact that anchor=2s did not, despite identical closeness=0.5).
    closeness and anchor_seconds were meant to be independent knobs (how
    strongly vs. where) but were actually entangled. Fixed by making
    closeness a fraction of a FIXED absolute pull-back distance
    (anchor_decouple_scale) instead of the variable, target-dependent one --
    now the same closeness means the same absolute coupling strength
    regardless of where anchor_seconds points.

    THREE REGIMES, confirmed by direct same-seed A/B testing (not just design
    intent) -- anchor_frame_index ("channel" in the frontend, deliberately
    not called a timestamp -- see MiniMaxH3TimelineEditor's own docstring)
    behaves as a channel-select for which references share one scene, not a
    timing control:
      - Unanchored (anchor_frame_index absent): the original, proven-clean
        default reference behavior -- present, but loose/non-dominant.
      - Two+ references anchored to the SAME target frame: bound together,
        genuinely co-occur in one shared scene/moment. This is the fix for
        the original "two references split into separate panels" problem.
      - Two+ references anchored to DIFFERENT target frames (even both
        otherwise valid, away from the clip's tail): confirmed to visibly
        drift apart into their own separate environments/moments instead of
        sharing a scene -- e.g. one reference's own photo background
        (a park) versus another's (a city skyscraper backdrop) each surfacing
        independently. Not yet exploited as a feature, but real: this is a
        plausible route to deliberately staging two characters in two
        different moments/scenes within one generation, if the prompt gives
        each one its own distinct described setting/action to reinforce it
        (untested pairing of channel-splitting with prompt-per-character
        specificity).
    The target frame's own magnitude barely matters within a regime (same-
    channel stayed stable and clean from ~0.1s to ~4.0s of a 5.167s clip);
    the one confirmed failure mode is anchoring very close to the clip's true
    final frame (~4.9s of 5.167s), which bled that reference's own photo
    background through hard right around that point.

    Bug-for-bug identical to native PackedLayout when refs is empty/None and
    no anchors are used."""
    frame, w_grid = h3model._frame_grid(latent_h, latent_w)
    frame_rows = frame.shape[0]
    target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
    if pretimeline_gap is None:
        pretimeline_gap = h3model.FRAME_RESCALE * 24.0
    if anchor_decouple_scale is None:
        anchor_decouple_scale = h3model.FRAME_RESCALE * 24.0 * 2.0

    # --- build reference segments first (buffered, not placed yet) so we
    # know where the video's real start time (video_origin) ends up, and can
    # retarget any anchored references against it afterward.
    ref_segments = []  # (kind, n_rows, is_audio)
    ref_pos = []
    image_anchor_targets = []  # (anchor_frame_index, pos_index, closeness)
    duration_anchor_targets = []  # (anchor_frame_index, [pos_indices], closeness, unanchored_start)
    cursor = float(text_len)
    if refs:
        for blk in refs:
            kind = blk["kind"]
            if kind == "image":
                r_frame, _ = h3model._frame_grid(blk["latent_h"], blk["latent_w"])
                n = r_frame.shape[0]
                g = torch.empty(n, 3, dtype=torch.float64)
                g[:, 0] = cursor
                g[:, 1:] = r_frame
                ref_segments.append(("ref_img", n, False))
                ref_pos.append(g)
                if blk.get("anchor_frame_index") is not None:
                    image_anchor_targets.append((blk["anchor_frame_index"], len(ref_pos) - 1, blk.get("anchor_closeness", 1.0)))
                cursor += 1.0
            elif kind == "audio":
                rt = blk["ref_audio_t"]
                block_start = cursor
                block_pos_indices = []
                if rt > 0:
                    ref_segments.append(("ref_audio", rt * 2, True))
                    ref_pos.append(h3model._audio_grid(cursor, rt, *target_audio_w))
                    block_pos_indices.append(len(ref_pos) - 1)
                if block_pos_indices and blk.get("anchor_frame_index") is not None:
                    duration_anchor_targets.append((blk["anchor_frame_index"], block_pos_indices, blk.get("anchor_closeness", 1.0), block_start))
                cursor += float(rt)
            elif kind in ("video", "video_audio"):
                rt = blk["ref_audio_t"]
                vt = blk["latent_t"]
                r_frame, r_w_grid = h3model._frame_grid(blk["latent_h"], blk["latent_w"])
                block_start = cursor
                block_pos_indices = []
                if rt > 0:
                    ref_segments.append(("ref_audio", rt * 2, True))
                    ref_pos.append(h3model._audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                    block_pos_indices.append(len(ref_pos) - 1)
                n = vt * r_frame.shape[0]
                ref_segments.append(("ref_img", n, False))
                ref_pos.append(h3model._video_grid(vt, r_frame, cursor))
                block_pos_indices.append(len(ref_pos) - 1)
                if blk.get("anchor_frame_index") is not None:
                    duration_anchor_targets.append((blk["anchor_frame_index"], block_pos_indices, blk.get("anchor_closeness", 1.0), block_start))
                cursor += max(float(rt), sum(h3model._video_t_spans(vt)))
    video_origin = cursor + float(pretimeline_gap)

    # --- retarget anchored references now that video_origin is known.
    token_times = h3model._video_t_grid(latent_t, 0.0)
    resolved = []
    for anchor_frame_index, pos_index, closeness in image_anchor_targets:
        token_index = min(_frame_index_to_token_index(anchor_frame_index), latent_t - 1)
        exact_t = video_origin + token_times[token_index].item()
        unanchored_t = ref_pos[pos_index][0, 0].item()
        closeness = max(0.0, min(1.0, float(closeness)))
        # FIXED distance (anchor_decouple_scale), not the variable distance to
        # exact_t -- see CLOSENESS BUG note above. closeness=1.0 always lands
        # exactly on exact_t; lower closeness pulls back by a constant amount
        # regardless of how far into the clip anchor_seconds points, so the
        # same closeness means the same absolute coupling strength everywhere.
        final_t = exact_t - (1.0 - closeness) * anchor_decouple_scale
        final_t = max(final_t, unanchored_t)
        ref_pos[pos_index][:, 0] = final_t
        resolved.append((pos_index, round(final_t, 3)))
    by_final_t = {}
    for pos_index, final_t in resolved:
        by_final_t.setdefault(final_t, []).append(pos_index)
    for pos_indices in by_final_t.values():
        for slot, pos_index in enumerate(pos_indices[1:], start=1):
            ref_pos[pos_index][:, 2] += slot * float(spatial_collision_offset)
    for anchor_frame_index, pos_indices, closeness, unanchored_start in duration_anchor_targets:
        token_index = min(_frame_index_to_token_index(anchor_frame_index), latent_t - 1)
        exact_start = video_origin + token_times[token_index].item()
        closeness = max(0.0, min(1.0, float(closeness)))
        # Same fixed-distance fix as the image case above.
        final_start = exact_start - (1.0 - closeness) * anchor_decouple_scale
        final_start = max(final_start, unanchored_start)
        delta = final_start - unanchored_start
        for pos_index in pos_indices:
            ref_pos[pos_index][:, 0] += delta

    # --- real assembly in native PackedLayout's physical order: text,
    # keyframe cond rows (anchored to video_origin, first/last or mid-clip),
    # reference rows (now at their retargeted positions), audio, video.
    segments = [("text", text_len)]
    g = torch.zeros(text_len, 3, dtype=torch.float64)
    g[:, 0] = torch.arange(text_len, dtype=torch.float64)
    pos = [g]
    img_pos, img_update = [], []
    audio_pos, audio_update = [], []
    row = text_len

    if keyframes:
        for kf in keyframes:
            pixel_index = kf["resolved_frame_index"]
            if pixel_index == 0:
                cond_t = video_origin
            elif frame_count is not None and pixel_index == frame_count - 1:
                cond_t = video_origin + sum(h3model._video_t_spans(latent_t)) - h3model.FRAME_RESCALE
            else:
                token_index = min(_frame_index_to_token_index(pixel_index), latent_t - 1)
                cond_t = video_origin + token_times[token_index].item()
            g = torch.empty(frame_rows, 3, dtype=torch.float64)
            g[:, 0] = cond_t
            g[:, 1:] = frame
            segments.append(("cond", frame_rows))
            pos.append(g)
            img_pos.append(torch.arange(row, row + frame_rows))
            img_update.append(torch.zeros(frame_rows, dtype=torch.bool))
            row += frame_rows

    for (kind, n, is_audio), p in zip(ref_segments, ref_pos):
        segments.append((kind, n))
        pos.append(p)
        if is_audio:
            audio_pos.append(torch.arange(row, row + n))
            audio_update.append(torch.zeros(n, dtype=torch.bool))
        else:
            img_pos.append(torch.arange(row, row + n))
            img_update.append(torch.zeros(n, dtype=torch.bool))
        row += n

    segments.append(("audio", audio_t * 2))
    pos.append(h3model._audio_grid(video_origin, audio_t, *target_audio_w))
    audio_pos.append(torch.arange(row, row + audio_t * 2))
    audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
    row += audio_t * 2

    n_video = latent_t * frame_rows
    segments.append(("video", n_video))
    pos.append(h3model._video_grid(latent_t, frame, video_origin))
    img_pos.append(torch.arange(row, row + n_video))
    img_update.append(torch.ones(n_video, dtype=torch.bool))
    row += n_video

    layout = h3model.PackedLayout.__new__(h3model.PackedLayout)
    layout.seq_len = row
    layout.position_ids = torch.cat(pos)
    layout.img_pos = torch.cat(img_pos) if img_pos else torch.zeros(0, dtype=torch.long)
    layout.img_update = torch.cat(img_update) if img_update else torch.zeros(0, dtype=torch.bool)
    layout.audio_pos = torch.cat(audio_pos) if audio_pos else torch.zeros(0, dtype=torch.long)
    layout.audio_update = torch.cat(audio_update) if audio_update else torch.zeros(0, dtype=torch.bool)
    layout.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
    seg_abs = []
    off = 0
    for kind, n in segments:
        seg_abs.append((off, off + n, kind))
        off += n
    layout.segments = seg_abs
    return layout


def _make_fixed_cond_video_rows(original_diffusion_model):
    """Per-item replacement for the DiT's own _cond_video_rows (comfy/ldm/
    minimax/model.py), which reads exactly ONE visual_cond_noise_aug scalar
    from the payload and applies it identically to every keyframe/reference
    row. This reads a PARALLEL cond_video_noise_augs list instead (built by
    _make_fixed_extra_conds, one entry per row, same order as
    cond_video_latents) so each row's own value is actually used -- falls
    back to the payload's single global value for any row past the end of
    that list (or if it's missing entirely), so this is safe to install even
    when nothing supplies per-item values.

    Deliberately does NOT patch the DiT's _forward (the ~130-line method
    that also decides each row's "apparent denoising timestep" via a
    per-KIND, not per-row, seg_t dict) -- that would mean duplicating a much
    larger, actively-maintained slice of the model's actual forward pass,
    a materially higher risk (silent drift on any core update, far larger
    surface for a subtle bug) than this narrow, well-scoped helper.
    Per-row noise-mixing (this) is the more direct lever on what content
    each row actually carries; the timestep-bucket piece staying per-kind
    is a deliberate scope limit, not an oversight."""
    def fixed_cond_video_rows(payload, device):
        rows = []
        latents = payload.get("cond_video_latents", [])
        default_aug = float(payload.get("visual_cond_noise_aug", h3model.VISUAL_COND_TIMESTEP))
        augs = payload.get("cond_video_noise_augs") or []
        seed = int(payload.get("seed", 0))
        for i, z in enumerate(latents):
            aug = float(augs[i]) if i < len(augs) else default_aug
            r = h3model.patchify_video(z.to(torch.float32), original_diffusion_model.patch_size)
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    return fixed_cond_video_rows


def _make_fixed_cond_audio_rows(original_diffusion_model):
    """Same per-item treatment as _make_fixed_cond_video_rows, for reference
    audio rows (cond_audio_latents / cond_audio_noise_augs)."""
    def fixed_cond_audio_rows(payload, device):
        rows = []
        latents = payload.get("cond_audio_latents", [])
        default_aug = float(payload.get("audio_cond_noise_aug", h3model.AUDIO_COND_TIMESTEP))
        augs = payload.get("cond_audio_noise_augs") or []
        seed = int(payload.get("seed", 0)) + 1
        for i, z in enumerate(latents):
            aug = float(augs[i]) if i < len(augs) else default_aug
            r = h3model.pack_audio(z.to(torch.float32))
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    return fixed_cond_audio_rows


def _make_fixed_extra_conds(original_model, pretimeline_gap_seconds, spatial_collision_offset, anchor_decouple_scale_seconds):
    """Returns a drop-in replacement for original_model.extra_conds, fixing
    the two verified bugs. Captures original_model in closure since
    add_object_patch stores this as a plain instance attribute -- called as
    self.extra_conds(**kwargs), which does NOT auto-bind self the way a
    class-level method would. pretimeline_gap/spatial_collision_offset/
    anchor_decouple_scale_seconds come from the Timeline Editor's own
    widgets, captured here so the layout builder (called per-generation,
    inside this closure) sees them."""

    def fixed_extra_conds(**kwargs):
        out = comfy.model_base.BaseModel.extra_conds(original_model, **kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            cross_attn = original_model.diffusion_model.preprocess_text_embeds(
                cross_attn.to(device=kwargs["device"], dtype=original_model.get_dtype_inference()))
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)

        latent_shapes = kwargs.get("latent_shapes", None)
        if latent_shapes is not None:
            out["latent_shapes"] = comfy.conds.CONDConstant(latent_shapes)

        payload = {}
        tags = kwargs.get("minimax_token_tags", None)
        if tags is not None:
            payload["text_token_tags"] = tags
        keyframes = kwargs.get("minimax_keyframes", None)
        refs = kwargs.get("minimax_refs", None)
        # THE FIX for bug 1: concatenate instead of the refs branch
        # unconditionally overwriting what the keyframes branch set.
        cond_video_latents = []
        # PER-ITEM noise_aug: a parallel list, same order/length as
        # cond_video_latents/cond_audio_latents, consumed by the patched
        # _cond_video_rows/_cond_audio_rows (see _make_fixed_cond_rows) --
        # native code only reads one global scalar per generation, this is
        # what makes each row's own value actually matter.
        cond_video_noise_augs = []
        if keyframes is not None:
            payload["keyframes"] = keyframes
            payload["frame_count"] = kwargs.get("minimax_frame_count", None)
            cond_video_latents.extend(kf["latent"] for kf in keyframes)
            cond_video_noise_augs.extend(kf.get("noise_aug", h3model.VISUAL_COND_TIMESTEP) for kf in keyframes)
        if refs is not None:
            payload["refs"] = refs
            cond_video_latents.extend(r["latent"] for r in refs if "latent" in r)
            cond_video_noise_augs.extend(r.get("noise_aug", h3model.VISUAL_COND_TIMESTEP) for r in refs if "latent" in r)
            payload["cond_audio_latents"] = [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
            payload["cond_audio_noise_augs"] = [r.get("noise_aug", h3model.AUDIO_COND_TIMESTEP) for r in refs if r.get("audio_latent") is not None]
        if keyframes is not None or refs is not None:
            payload["cond_video_latents"] = cond_video_latents
            payload["cond_video_noise_augs"] = cond_video_noise_augs
        if kwargs.get("minimax_visual_cond_noise_aug", None) is not None:
            payload["visual_cond_noise_aug"] = kwargs["minimax_visual_cond_noise_aug"]
        if kwargs.get("minimax_audio_cond_noise_aug", None) is not None:
            payload["audio_cond_noise_aug"] = kwargs["minimax_audio_cond_noise_aug"]
        payload["seed"] = kwargs.get("seed", 0)
        payload["audio_scale"] = original_model.audio_scale()
        if cross_attn is not None and latent_shapes is not None and len(latent_shapes) > 1:
            vs = latent_shapes[0]
            # THE FIX for bug 2: our corrected layout builder, not the
            # native PackedLayout, so a keyframe's anchor accounts for
            # however far references push the real video origin.
            payload["layout"] = _corrected_packed_layout(
                cross_attn.shape[1], vs[2], (vs[3] + 1) // 2 * 2, (vs[4] + 1) // 2 * 2,
                latent_shapes[1][-1], keyframes=payload.get("keyframes"),
                refs=payload.get("refs"), frame_count=payload.get("frame_count"),
                pretimeline_gap=h3model.FRAME_RESCALE * 24.0 * pretimeline_gap_seconds,
                spatial_collision_offset=spatial_collision_offset,
                anchor_decouple_scale=h3model.FRAME_RESCALE * 24.0 * anchor_decouple_scale_seconds)
        out["minimax_payload"] = comfy.conds.CONDConstant(payload)
        return out

    return fixed_extra_conds


def _anchor_frame_index(item: _TimelineItem, frame_count: int) -> int | None:
    if item.anchor_seconds < 0.0:
        return None
    return max(0, min(frame_count - 1, round(item.anchor_seconds * h3.FPS)))


def _combined_conditioning(clip, video_vae, audio_vae, prompt, width, height, length, ref_image_size, timeline: MiniMaxH3TimelineBundle):
    items = timeline.items
    keyframe_start = next((i for i in items if i.role == KEYFRAME_START), None)
    keyframe_end = next((i for i in items if i.role == KEYFRAME_END), None)
    keyframe_mids = [i for i in items if i.role == KEYFRAME_MID]
    reference_items = [i for i in items if i.role == REFERENCE]

    latent, frame_count = h3._empty_av_latent(width, height, length)

    # --- keyframes (adapted from _empty_image_conditioning) ---
    keyframe_images = []
    keyframes = []
    if keyframe_start is not None:
        if keyframe_start.media_type != "image":
            raise ValueError("keyframe_start must be an image")
        source = _load_media_file(keyframe_start.filename, "image")[:1]
        image = h3._resize(source, width, height, "center")
        keyframe_images.append(image)
        keyframes.append({"resolved_frame_index": 0, "image": image, "noise_aug": keyframe_start.noise_aug})
    if keyframe_end is not None:
        if keyframe_end.media_type != "image":
            raise ValueError("keyframe_end must be an image")
        source = _load_media_file(keyframe_end.filename, "image")[:1]
        image = h3._resize(source, width, height, "center")
        keyframe_images.append(image)
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": image, "noise_aug": keyframe_end.noise_aug})
    for item in keyframe_mids:
        if item.media_type != "image":
            raise ValueError("keyframe_mid items must be images")
        source = _load_media_file(item.filename, "image")[:1]
        image = h3._resize(source, width, height, "center")
        keyframe_images.append(image)
        keyframes.append({"resolved_frame_index": _anchor_frame_index(item, frame_count), "image": image, "noise_aug": item.noise_aug})
    for kf in keyframes:
        kf["latent"] = video_vae.encode(kf.pop("image"))

    # --- references (adapted from _reference_conditioning) ---
    ref_items: list[dict] = []
    ref_blocks: list[dict] = []
    tag_by_input: dict[int, str] = {}
    soundtrack_pairs: list[tuple[int, int]] = []
    images = [i for i in reference_items if i.media_type == "image"]
    videos = [i for i in reference_items if i.media_type == "video"]
    audios = [i for i in reference_items if i.media_type == "audio"]
    audio_ordinal = 0

    for picture_ordinal, item in enumerate(images, start=1):
        image = _load_media_file(item.filename, "image")
        image_h, image_w = image.shape[1], image.shape[2]
        size_mode = str(ref_image_size or REF_IMAGE_1K)
        if size_mode == REF_IMAGE_ORIGINAL:
            target_w, target_h = _original_reference_size(image_w, image_h)
            if target_w == image_w and target_h == image_h:
                resized = image[:1]
            elif image_w >= h3.CANVAS_MULTIPLE and image_h >= h3.CANVAS_MULTIPLE:
                top = (image_h - target_h) // 2
                left = (image_w - target_w) // 2
                resized = image[:1, top:top + target_h, left:left + target_w, :]
            else:
                resized = h3._resize(image[:1], target_w, target_h, "disabled")
            z = video_vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({
                "kind": "image", "latent_h": int(z.shape[-2]), "latent_w": int(z.shape[-1]), "latent": z,
                "anchor_frame_index": _anchor_frame_index(item, frame_count), "anchor_closeness": 1.0, "noise_aug": item.noise_aug,
            })
            tag_by_input[item.item_index] = f"<Picture {picture_ordinal}>"
            continue
        if size_mode == REF_IMAGE_MATCH:
            target_area = width * height
        else:
            target_area = REFERENCE_IMAGE_AREAS.get(size_mode, REFERENCE_IMAGE_AREAS[REF_IMAGE_1K])
        scale = min(1.0, math.sqrt(target_area / max(1, image_w * image_h)))
        target_w, target_h = _reference_aligned_size(image_w, image_h, scale)
        resized = h3._resize(image[:1], target_w, target_h, "disabled")
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({
            "kind": "image", "latent_h": target_h // 16, "latent_w": target_w // 16, "latent": video_vae.encode(resized),
            "anchor_frame_index": _anchor_frame_index(item, frame_count), "anchor_closeness": 1.0, "noise_aug": item.noise_aug,
        })
        tag_by_input[item.item_index] = f"<Picture {picture_ordinal}>"

    for video_ordinal, item in enumerate(videos, start=1):
        frames, soundtrack, source_fps = _video_parts(_load_media_file(item.filename, "video"))
        frames = _resample_video_frames(frames, source_fps)
        video_h, video_w = frames.shape[1], frames.shape[2]
        canvas_w, canvas_h = h3.adapt_canvas(video_w, video_h)
        if video_w * video_h < canvas_w * canvas_h:
            canvas_w = max(h3.CANVAS_MULTIPLE, round(video_w / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
            canvas_h = max(h3.CANVAS_MULTIPLE, round(video_h / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        frames = h3._resize(frames, canvas_w, canvas_h, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        count = frames.shape[0]
        if count < 5:
            raise ValueError("Reference videos need at least 5 frames")
        while count % 17 != 5:
            count -= 1
        frames = frames[:count]
        video_latent = video_vae.encode(frames)
        audio_latent = None
        audio_t = 0
        if soundtrack is not None:
            audio_latent, audio_t = _encode_reference_audio(audio_vae, soundtrack)
            audio_ordinal += 1
            soundtrack_pairs.append((audio_ordinal, video_ordinal))
            ref_items.append({"type": "audio"})
        sample_indexes = list(range(0, frames.shape[0], h3.FPS // 2))
        ref_items.append({"type": "video", "data": frames[sample_indexes], "timestamps": [i / 2.0 for i in range(len(sample_indexes))]})
        ref_blocks.append({
            "kind": "video_audio" if audio_t else "video",
            "latent_t": video_latent.shape[2], "latent_h": canvas_h // 16, "latent_w": canvas_w // 16,
            "ref_audio_t": audio_t, "latent": video_latent, "audio_latent": audio_latent,
            "anchor_frame_index": _anchor_frame_index(item, frame_count), "anchor_closeness": 1.0, "noise_aug": item.noise_aug,
        })
        tag_by_input[item.item_index] = f"<Video {video_ordinal}>"

    for item in audios:
        audio = _load_media_file(item.filename, "audio")
        audio_latent, audio_t = _encode_reference_audio(audio_vae, audio)
        audio_ordinal += 1
        ref_items.append({"type": "audio"})
        ref_blocks.append({
            "kind": "audio", "ref_audio_t": audio_t, "audio_latent": audio_latent,
            "anchor_frame_index": _anchor_frame_index(item, frame_count), "anchor_closeness": 1.0, "noise_aug": item.noise_aug,
        })
        tag_by_input[item.item_index] = f"<Audio {audio_ordinal}>"

    if reference_items and (not ref_items or all(i.get("type") == "audio" for i in ref_items)):
        raise ValueError("Reference items need at least one image or video")

    resolved_prompt = _resolve_reference_prompt(prompt, tag_by_input, soundtrack_pairs, len(videos), len(audios)) if reference_items else prompt

    tokenize_kwargs = {}
    if keyframe_images:
        tokenize_kwargs["images"] = keyframe_images
    if ref_items:
        tokenize_kwargs["minimax_ref_items"] = ref_items
    tokens = clip.tokenize(resolved_prompt, **tokenize_kwargs)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if keyframes:
        conditioning = node_helpers.conditioning_set_values(conditioning, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })
    if ref_blocks:
        conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
    if keyframes or ref_blocks:
        # Native MiniMax H3 parameters -- see MiniMaxH3TimelineBundle's
        # docstring comment. _make_fixed_extra_conds already forwards these
        # from kwargs into the payload if present; this is what actually
        # supplies them, which nothing did before.
        conditioning = node_helpers.conditioning_set_values(conditioning, {
            "minimax_visual_cond_noise_aug": timeline.visual_cond_noise_aug,
            "minimax_audio_cond_noise_aug": timeline.audio_cond_noise_aug,
        })
    return conditioning, latent


class MiniMaxH3ConditioningTimelineIntegration:
    """Consumes a `timeline` bundle plus a MODEL/CLIP/VAE/VAE wired directly
    from native loader nodes (Load Diffusion Model, Load CLIP with
    type=minimax, Load VAE x2 -- NOT MiniMax H3 Easy Loader's bundle), and
    builds ONE conditioning object carrying both keyframes and references
    together -- the combination native ComfyUI cannot correctly combine (see
    module docstring for the two bugs and their fix).

    Deliberately takes raw native-loader connections instead of a bundle:
    MiniMaxH3Bundle.model_for() silently substitutes whichever H3 checkpoint
    IS configured when the one a mode prefers is left unset ("None"), so
    which checkpoint a generation actually runs on can differ from what a
    dropdown shows, and only becomes visible after the fact (in the console
    log). Wiring a specific Load Diffusion Model node's output directly
    means there is nothing to substitute -- what's connected on the canvas,
    before you run anything, IS what gets used.

    video_vae/audio_vae are REQUIRED INPUTS (this node uses them internally
    to encode keyframe/reference media into latents) but are deliberately
    NOT re-emitted as outputs. Only `model` needs to flow out of this node --
    it may be a cloned, patched object, not the same one that came in (see
    below) -- so it's the only thing downstream nodes are forced to route
    through here. VAEDecode/VAEDecodeAudio should wire directly to the same
    Load VAE nodes connected here instead of through this node's output,
    which used to exist purely as a passthrough and just added an unneeded
    dependency for anyone whose graph already wires VAE elsewhere."""

    CATEGORY = "MiniMax H3 Easy/Timeline"
    FUNCTION = "generate"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "latent", "fps")
    DESCRIPTION = "Builds combined keyframe+reference conditioning from a Timeline Editor bundle. Connect a MODEL from Load Diffusion Model, a CLIP from Load CLIP (type=minimax), and two VAEs from Load VAE -- not MiniMax H3 Easy Loader's bundle -- so the checkpoint in use is whatever's visibly wired in, not resolved silently at runtime. Wire VAEDecode/VAEDecodeAudio directly to the same Load VAE nodes, not through this node's output -- only `model` needs to come from here."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "timeline": ("MINIMAX_H3_TIMELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (list(RESOLUTION_MEGAPIXELS.keys()), {"default": RESOLUTION_480}),
                "aspect_ratio": (list(ASPECT_RATIOS.keys()), {"default": ASPECT_WIDESCREEN}),
                "ref_image_size": (["match", "1k", "1.5k", "2k", "original"], {"default": "match"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
        }

    def generate(self, model, clip, video_vae, audio_vae, timeline, prompt, resolution, aspect_ratio, ref_image_size, fps=24.0):
        if not isinstance(timeline, MiniMaxH3TimelineBundle):
            raise ValueError("Connect a MiniMax H3 Timeline Editor output")
        has_keyframe = any(i.role in KEYFRAME_ROLES for i in timeline.items)
        has_reference = any(i.role == REFERENCE for i in timeline.items)
        has_mid_keyframe = any(i.role == KEYFRAME_MID for i in timeline.items)

        width, height = _canvas_dimensions(resolution, aspect_ratio, 0, 0)
        length = _frame_length(timeline.duration_seconds, h3.FPS)

        conditioning, latent = _combined_conditioning(clip, video_vae, audio_vae, prompt, width, height, length, ref_image_size, timeline)

        # The patched layout builder is needed whenever native PackedLayout
        # can't correctly express what's being asked: the keyframe+reference
        # origin bug (bug 1+2, see module docstring), or any mid-clip keyframe
        # (native only knows first/last). References no longer anchor, so
        # that's no longer a trigger condition here.
        needs_layout_patch = (has_keyframe and has_reference) or has_mid_keyframe
        # Per-item noise_aug needs to apply to a pure-keyframe or
        # pure-reference generation too, not just combined ones -- separate
        # trigger from the layout patch above.
        needs_noise_aug_patch = has_keyframe or has_reference
        if needs_layout_patch or needs_noise_aug_patch:
            model = model.clone()
            original_model = model.model
            if needs_layout_patch:
                model.add_object_patch(
                    "extra_conds",
                    _make_fixed_extra_conds(
                        original_model, timeline.pretimeline_gap_seconds, timeline.spatial_collision_offset,
                        timeline.anchor_decouple_scale_seconds,
                    ),
                )
            if needs_noise_aug_patch:
                # add_object_patch supports dotted paths (comfy.utils.set_attr/
                # resolve_attr), so this reaches the DiT instance nested inside
                # the model directly.
                original_diffusion_model = original_model.diffusion_model
                model.add_object_patch("diffusion_model._cond_video_rows", _make_fixed_cond_video_rows(original_diffusion_model))
                model.add_object_patch("diffusion_model._cond_audio_rows", _make_fixed_cond_audio_rows(original_diffusion_model))

        return (model, conditioning, latent, float(fps))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TimelineEditor": MiniMaxH3TimelineEditor,
    "MiniMaxH3ConditioningTimelineIntegration": MiniMaxH3ConditioningTimelineIntegration,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TimelineEditor": "MiniMax H3 Timeline Editor",
    "MiniMaxH3ConditioningTimelineIntegration": "MiniMax H3 Conditioning (Timeline Integration)",
}
