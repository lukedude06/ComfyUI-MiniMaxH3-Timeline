"""MiniMax H3 Timeline Editor + combined keyframe/reference conditioning.

Nodes:
  MiniMaxH3TimelineEditor                   cards -> a timeline bundle
  MiniMaxH3ConditioningTimelineIntegration  bundle -> conditioning + latent + fps
  MiniMaxH3TimelineModelPatch               MODEL -> patched MODEL
  MiniMaxH3TextEncoderLoader                Load CLIP with a config override

THE ORDERING CONTRACT -- the invariant everything else rests on. Three
structures must stay in lockstep, all of them keyframes-then-references:

  1. cond_video_latents / cond_audio_latents and their parallel
     cond_video_noise_augs / cond_audio_noise_augs, built in fixed_extra_conds
  2. the cond / cond_audio / ref_img / ref_audio segments emitted by
     _corrected_packed_layout
  3. the order those aug lists are consumed in fixed_forward

Nothing validates this. Desynchronise any one of them and rows silently take
another row's content or apparent timestep, with no error.

The patches are installed via model.clone().add_object_patch(...) and are
stored as plain instance attributes, not bound methods -- so each factory
captures the model it was built for, because `self` is not supplied at call
time. They have no static callers in this file by design; ComfyUI invokes
them on the patched model during sampling.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Mapping

import folder_paths
import torch
import torchaudio

import node_helpers
import nodes
import comfy.conds
# common_dit / model_prefetch are used inside the patched _forward. They used to
# resolve only because ComfyUI core happens to import them first; import them
# here so this file does not depend on someone else's import order.
import comfy.ldm.common_dit
import comfy.model_base
import comfy.model_management
import comfy.model_prefetch
import comfy.nested_tensor
import comfy.sd
import comfy.utils
from comfy_api.latest import InputImpl
from comfy_extras.nodes_audio import load as _load_audio_waveform

# The MiniMax H3 packed-sequence math this pack patches (PackedLayout, the
# position grids, patchify/pack, the RoPE table, the cond-noise-aug constants)
# is used from a PINNED vendored snapshot so a ComfyUI update cannot silently
# change it under the timeline logic. Fall back to the live modules if the
# snapshot ever fails to import against a future core.
try:
    from ._vendor import mmx_model as h3model
    from ._vendor import mmx_extras as h3
    _MMX_VENDORED = True
except Exception:  # pragma: no cover - defensive fallback
    import comfy.ldm.minimax.model as h3model  # type: ignore
    from comfy_extras import nodes_minimax_h3 as h3  # type: ignore
    _MMX_VENDORED = False

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
    # Global fallbacks. build_timeline resolves a per-item noise_aug for every
    # card, so the parallel lists handed to the patches are always fully
    # populated and these are never read in normal use. Kept for an item that
    # carries no noise_aug field at all.
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

    CATEGORY = "MiniMax H3 Timeline"
    FUNCTION = "build_timeline"
    RETURN_TYPES = ("MINIMAX_H3_TIMELINE",)
    RETURN_NAMES = ("timeline",)
    DESCRIPTION = "Upload media and mark each item as a keyframe (start/end/mid) or a reference. Wire into MiniMax H3 Conditioning (Timeline Integration)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 15.0, "step": 0.1}),
                # These two globals are effectively dead in normal use --
                # every card supplies its own per-item noise_aug, so these are
                # only a fallback for an item with no noise_aug field at all.
                # 0.999 / 1.0 are the per-modality defaults: how resolved a row
                # is treated as from the first sampling step.
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
                              frame_count=None, audio_keyframes=None, pretimeline_gap=None,
                              spatial_collision_offset=64.0, anchor_decouple_scale=None):
    """PackedLayout built for what the timeline needs to express.

    Emits, in physical order: text, keyframe cond rows, audio-keyframe
    cond_audio rows, reference rows, target audio, target video.

    Reference segments are buffered before anything is placed, so the cursor's
    final position is known up front; video_origin is that cursor plus
    pretimeline_gap, and keyframe anchors are computed against video_origin.
    A keyframe's rotary time is the real time of the compression token
    containing its frame (see _frame_index_to_token_index), not a linear
    frame offset.

    Handles multi-frame keyframes (latent_t > 1) and the cond_audio segment
    kind for audio pinned into the target's own audio track.

    Per-reference anchoring (anchor_frame_index / anchor_closeness,
    spatial_collision_offset, anchor_decouple_scale) is implemented here but
    DORMANT: build_timeline forces references to ANCHOR_UNSET, so
    _anchor_frame_index returns None and these branches never run. Intact
    rather than deleted -- it re-arms if a reference ever carries a real
    anchor_seconds again.
    """
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
            # Video-keyframe path (see _combined_conditioning):
            # a multi-frame keyframe has no single-frame closed-form shortcut
            # to reuse, so it always goes through the general token-index
            # lookup below, using the clip's START frame -- for the "end"
            # role, that's backdated so the clip's LAST frame lands on
            # frame_count - 1 instead of its first.
            vt = kf.get("latent_t", 1)
            if vt > 1:
                start_pixel_index = max(0, pixel_index - (vt - 1)) if pixel_index == frame_count - 1 else pixel_index
                token_index = min(_frame_index_to_token_index(start_pixel_index), latent_t - 1)
                cond_t = video_origin + token_times[token_index].item()
                g = h3model._video_grid(vt, frame, cond_t)
            elif pixel_index == 0:
                cond_t = video_origin
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
            elif frame_count is not None and pixel_index == frame_count - 1:
                cond_t = video_origin + sum(h3model._video_t_spans(latent_t)) - h3model.FRAME_RESCALE
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
            else:
                token_index = min(_frame_index_to_token_index(pixel_index), latent_t - 1)
                cond_t = video_origin + token_times[token_index].item()
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
            n = frame_rows * vt
            segments.append(("cond", n))
            pos.append(g)
            img_pos.append(torch.arange(row, row + n))
            img_update.append(torch.zeros(n, dtype=torch.bool))
            row += n

    # Audio keyframes -- fixed audio content pinned to a
    # specific point in the TARGET's own audio track (a new "cond_audio"
    # segment kind), not an independent pretimeline reference block like
    # ref_audio. Reuses _audio_grid exactly like ref_audio does, but
    # anchored at video_origin + an audio-latent-frame offset instead of
    # the pretimeline cursor -- same trick the video-keyframe path above
    # uses (an existing reference-block position builder, repositioned onto
    # the target's own timeline). Audio position is linear (1 unit per
    # audio-latent-frame, no per-cycle rescaling like video needs), so no
    # token-index lookup table is needed here, unlike the video case.
    if audio_keyframes:
        for akf in audio_keyframes:
            audio_index = akf["resolved_audio_frame_index"]
            rt = akf.get("latent_t", 1)
            if audio_t is not None and audio_index == audio_t - 1 and rt > 1:
                start_audio_index = max(0, audio_index - (rt - 1))
            else:
                start_audio_index = max(0, min(audio_t - 1, audio_index)) if audio_t is not None else audio_index
            cond_audio_t = video_origin + start_audio_index
            n = rt * 2
            segments.append(("cond_audio", n))
            pos.append(h3model._audio_grid(cond_audio_t, rt, *target_audio_w))
            audio_pos.append(torch.arange(row, row + n))
            audio_update.append(torch.zeros(n, dtype=torch.bool))
            row += n

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

    On its own this only controls how much noise gets mixed into a row's
    actual content. The DiT's _forward also derives each row's "apparent
    denoising timestep" (seg_t, per-KIND not per-row) from the SAME global
    scalar, which is a separate signal (how much the model trusts/commits to
    that row) that happens to be tied to the same number. Lowering noise_aug
    to soften a hard cut therefore also always lowers that trust signal for
    every row of that kind -- see _make_fixed_forward, which decouples the
    two by making seg_t per-row too, so a row's content and its apparent
    timestep can be set independently."""
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


def _final_layer_takes_sample_args(final_layer):
    """True if the installed FinalLayer.forward wants the post-v0.34.2
    (sigma, sample_sigmas, shifts) trailing args. Result cached on the class.
    Falls back to a positional-count check if the signature cannot be read."""
    cls = type(final_layer)
    cached = getattr(cls, "_mmx_timeline_fl_extra", None)
    if cached is not None:
        return cached
    result = False
    try:
        params = inspect.signature(cls.forward).parameters
        if {"sigma", "sample_sigmas", "shifts"} & set(params):
            result = True
        elif not any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()):
            positional = [p for p in params.values()
                          if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                        inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            result = len(positional) > 5  # self, x, t_emb, video_seg, audio_seg
    except (TypeError, ValueError):
        result = False
    try:
        cls._mmx_timeline_fl_extra = result
    except Exception:
        pass
    return result


# Attributes/methods the vendored _forward / _cond_*_rows / extra_conds patches
# read off the LIVE (ComfyUI-built) diffusion-model instance. These are
# checkpoint-frozen, so drift is rare -- but if a ComfyUI refactor renames or
# drops one, apply nothing and say so loudly instead of producing wrong output.
_REQUIRED_DIT_ATTRS = (
    "patch_size", "latents_dim", "hidden_size",
    "sigma_shift_video", "sigma_shift_audio", "use_adaln_curves",
    "video_patch_proj", "audio_patch_proj", "condition_proj", "token_refiner",
    "blocks", "final_layer", "rope_freqs",
    "_cond_video_rows", "_cond_audio_rows", "_forward",
)


def _dit_compatibility_report(model_patcher):
    """('' if the live DiT instance exposes everything the timeline patches
    need, else a human-readable list of what's missing)."""
    diffusion_model = getattr(getattr(model_patcher, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        return "diffusion_model (not a MiniMax H3 model?)"
    missing = [a for a in _REQUIRED_DIT_ATTRS if not hasattr(diffusion_model, a)]
    if not diffusion_model.__dict__.get("use_adaln_curves", getattr(diffusion_model, "use_adaln_curves", None)):
        if not hasattr(diffusion_model, "time_embedder"):
            missing.append("time_embedder")
    elif not hasattr(diffusion_model, "adaln_t_table"):
        missing.append("adaln_t_table")
    return ", ".join(missing)


def _make_fixed_forward(original_diffusion_model):
    """Replacement for the DiT's own _forward (comfy/ldm/minimax/model.py)
    that makes the seg_t "apparent denoising timestep" per-ROW instead of
    per-KIND. Native code computes ONE seg_t["cond"]/seg_t["ref_img"]/
    seg_t["ref_audio"] value from the single global noise_aug scalar and
    applies it to every keyframe/reference row of that kind; this instead
    walks the same cond_video_noise_augs/cond_audio_noise_augs parallel
    lists _make_fixed_cond_video_rows/_make_fixed_cond_audio_rows already
    use, in the same order layout.segments emits "cond"/"ref_img"/
    "ref_audio" segments (keyframes then refs, matching how
    _make_fixed_extra_conds built those lists), so each row's own value
    drives its own timestep bucket.

    This is otherwise a faithful copy of the native method -- everything
    past the seg_t/mod_segments construction (embedding, AdaLN table
    lookup, RoPE, the block loop, final_layer) is untouched. Duplicating
    this much of the forward pass is a real, accepted maintenance cost:
    it will silently drift from any future upstream change to _forward
    until this file is updated to match. Where a drift has a stable,
    detectable shape it is adapted at runtime instead -- e.g. the
    final_layer call below, whose signature changed after ComfyUI
    v0.34.2 (see _final_layer_takes_sample_args). Captures
    original_diffusion_model
    in closure since add_object_patch stores this as a plain instance
    attribute, not a bound method -- called as self._forward(...) would
    NOT auto-bind self the way a class-level method does."""
    def fixed_forward(x, timestep, context, transformer_options={}, minimax_payload=None,
                      denoise_mask=None, audio_denoise_mask=None, **kwargs):
        m = original_diffusion_model
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, m.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]
        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = h3model.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                           keyframes=payload.get("keyframes"),
                                           refs=payload.get("refs"),
                                           frame_count=payload.get("frame_count"))

        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", m.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", m.sigma_shift_audio))
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - h3model.time_shift_sigma(sigma_v, shift_v, shift_a))

        # per-ROW apparent timestep, unlike native's per-KIND seg_t dict --
        # walk layout.segments in order, consuming cond_video_noise_augs for
        # each "cond"/"ref_img" segment and cond_audio_noise_augs for each
        # "ref_audio" segment (same order _make_fixed_extra_conds built
        # them: keyframes then refs), falling back to the payload's single
        # global scalar past the end of either list.
        default_vis_aug = float(payload.get("visual_cond_noise_aug", h3model.VISUAL_COND_TIMESTEP))
        default_aud_aug = float(payload.get("audio_cond_noise_aug", h3model.AUDIO_COND_TIMESTEP))
        vis_augs = payload.get("cond_video_noise_augs") or []
        aud_augs = payload.get("cond_audio_noise_augs") or []
        # DENOISE MASKS -- ported from core's _forward. A masked row runs at
        # sigma = m * sigma_stream, so its label is 1 - m*sigma, clamped at the
        # cond timestep for a fully preserved row. Without this the model is
        # never told a frozen region is already resolved and treats it as noise
        # at the current sigma, which is exactly the signal any tiled/inpaint
        # workflow depends on. Kept per-ROW here, same as the aug handling below.
        t_pin_v = max(t_v, h3model.VISUAL_COND_TIMESTEP)
        t_pin_a = max(t_a, h3model.AUDIO_COND_TIMESTEP)
        video_seg_t, audio_seg_t = t_v, t_a
        video_rows_t = audio_rows_t = None
        # NB: `m` is the diffusion model in this scope (m = original_diffusion_model
        # above), unlike core's _forward where it is free -- so the mask locals
        # are named mask_vals here. Shadowing it breaks every later m.<attr>.
        if denoise_mask is not None:
            mask_vals = h3model.mask_row_values(
                denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
            if mask_vals is not None:
                rows_t = (1.0 - mask_vals * sigma_v.to(mask_vals.device)).clamp(max=t_pin_v)
                if rows_t.unique().numel() == 1:
                    video_seg_t = float(rows_t[0])
                else:
                    video_rows_t = rows_t
        if audio_denoise_mask is not None:
            mask_vals = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
            if not bool((mask_vals >= 1.0 - 1e-3).all()):
                sigma_a = 1.0 - t_a
                rows_t = (1.0 - mask_vals * sigma_a).clamp(max=t_pin_a)
                if rows_t.unique().numel() == 1:
                    audio_seg_t = float(rows_t[0])
                else:
                    audio_rows_t = rows_t

        vis_idx = aud_idx = 0
        seg_t_list = []
        for a, b, kind in layout.segments:
            if kind == "video":
                seg_t_list.append(video_seg_t)
            elif kind == "audio":
                seg_t_list.append(audio_seg_t)
            elif kind in ("cond", "ref_img"):
                aug = float(vis_augs[vis_idx]) if vis_idx < len(vis_augs) else default_vis_aug
                vis_idx += 1
                seg_t_list.append(max(t_v, aug))
            elif kind in ("ref_audio", "cond_audio"):
                # cond_audio is a new segment kind (an audio
                # keyframe pinned to a specific point in the target's own
                # audio track, see _corrected_packed_layout) -- shares the
                # same aud_augs list/consumption order as ref_audio since
                # _make_fixed_extra_conds builds cond_audio_latents/
                # cond_audio_noise_augs from keyframe audio THEN ref audio,
                # matching the physical segment order the layout emits them.
                aug = float(aud_augs[aud_idx]) if aud_idx < len(aud_augs) else default_aud_aug
                aud_idx += 1
                seg_t_list.append(max(t_a, aug))
            else:  # text
                seg_t_list.append(t_v)
        unique_t = sorted(set(seg_t_list) | {t_v, t_a}
                          | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
                          | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set()))
        t_row = {t: i for i, t in enumerate(unique_t)}

        def rows_to_mod_index(rows_t, tag):
            """per-row timestep values -> per-row mod-row indices into t_emb"""
            levels = rows_t.unique()
            base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()],
                                dtype=torch.long, device=rows_t.device)
            return base[torch.searchsorted(levels, rows_t)]
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2, "cond_audio": 2}

        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for seg_idx, (a, b, kind) in enumerate(layout.segments):
            row_base = t_row[seg_t_list[seg_idx]] * 3
            if kind == "text" and text_tags is not None:
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            elif kind == "video" and video_rows_t is not None:
                mod_segments.append((a, b, rows_to_mod_index(video_rows_t, seg_tag[kind])))
            elif kind == "audio" and audio_rows_t is not None:
                mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, seg_tag[kind])))
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = h3model.patchify_video(video_x.to(torch.float32), m.patch_size)
        audio_rows = h3model.pack_audio(audio_x.to(torch.float32))
        cond_video_rows = m._cond_video_rows(payload, device)
        cond_audio_rows = m._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = m.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = m.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != m.hidden_size:
            text_states = m.token_refiner(m.condition_proj(text_states),
                                          transformer_options=transformer_options)

        h = torch.empty(layout.seq_len, m.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if m.use_adaln_curves:
            table = comfy.model_management.cast_to(m.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = m.time_embedder(t_vals).to(dtype)

        rope_freqs = h3model.rope_rotation_table(m.rope_freqs(layout.position_ids, device), dtype)

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(m.blocks), device, transformer_options)
        for i, block in enumerate(m.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                         transformer_options=args["transformer_options"])}
                h = blocks_replace[("double_block", i)](
                    {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                     "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

        va, vb, _ = next(sg for sg in layout.segments if sg[2] == "video")
        aa, ab, _ = next(sg for sg in layout.segments if sg[2] == "audio")
        video_seg = ((va, vb, rows_to_mod_index(video_rows_t, 0) // 3) if video_rows_t is not None
                     else (va, vb, t_row[video_seg_t]))
        audio_seg = ((aa, ab, rows_to_mod_index(audio_rows_t, 0) // 3) if audio_rows_t is not None
                     else (aa, ab, t_row[audio_seg_t]))
        # final_layer.forward gained (sigma, sample_sigmas, shifts) after
        # ComfyUI v0.34.2 (PDD-head banking). Adapt to whichever signature is
        # installed so this pack keeps working across ComfyUI versions -- every
        # value it wants is already computed above.
        if _final_layer_takes_sample_args(m.final_layer):
            v, a = m.final_layer(h, t_emb, video_seg, audio_seg,
                                 sigma_v,
                                 transformer_options.get("sample_sigmas"),
                                 (shift_v, shift_a))
        else:
            v, a = m.final_layer(h, t_emb, video_seg, audio_seg)

        video_out = h3model.unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, m.latents_dim, m.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = h3model.unpack_audio(a)

        return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]

    return fixed_forward


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
        # EARLY-OUT: nothing from a Timeline Editor in this conditioning, so
        # behave exactly like native. Without this, a model patched by the
        # standalone MiniMaxH3TimelineModelPatch node would shift every plain
        # t2v/fl2va/ref2va generation by pretimeline_gap, because
        # _corrected_packed_layout adds that gap unconditionally while native
        # PackedLayout puts video_origin straight at the post-reference cursor.
        # Gating here rather than at graph-build time is what makes the patch
        # node safe to leave permanently in the model chain.
        if (kwargs.get("minimax_keyframes") is None
                and kwargs.get("minimax_refs") is None
                and kwargs.get("minimax_audio_keyframes") is None):
            return comfy.model_base.MiniMaxH3.extra_conds(original_model, **kwargs)

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
        # Audio keyframes: same idea as video keyframes, but for
        # the "cond_audio" segment kind (see _corrected_packed_layout and
        # _make_fixed_forward's seg_tag/seg_t_list handling). Built into
        # cond_audio_latents/cond_audio_noise_augs BEFORE ref-derived audio,
        # matching the physical segment order _corrected_packed_layout emits
        # them (keyframe audio segments come before reference segments).
        audio_keyframes = kwargs.get("minimax_audio_keyframes", None)
        cond_audio_latents = []
        cond_audio_noise_augs = []
        if audio_keyframes is not None:
            payload["audio_keyframes"] = audio_keyframes
            cond_audio_latents.extend(akf["latent"] for akf in audio_keyframes)
            cond_audio_noise_augs.extend(akf.get("noise_aug", h3model.AUDIO_COND_TIMESTEP) for akf in audio_keyframes)
        if refs is not None:
            payload["refs"] = refs
            cond_video_latents.extend(r["latent"] for r in refs if "latent" in r)
            cond_video_noise_augs.extend(r.get("noise_aug", h3model.VISUAL_COND_TIMESTEP) for r in refs if "latent" in r)
            cond_audio_latents.extend(r["audio_latent"] for r in refs if r.get("audio_latent") is not None)
            cond_audio_noise_augs.extend(r.get("noise_aug", h3model.AUDIO_COND_TIMESTEP) for r in refs if r.get("audio_latent") is not None)
        if keyframes is not None or refs is not None:
            payload["cond_video_latents"] = cond_video_latents
            payload["cond_video_noise_augs"] = cond_video_noise_augs
        if audio_keyframes is not None or refs is not None:
            payload["cond_audio_latents"] = cond_audio_latents
            payload["cond_audio_noise_augs"] = cond_audio_noise_augs
        if kwargs.get("minimax_visual_cond_noise_aug", None) is not None:
            payload["visual_cond_noise_aug"] = kwargs["minimax_visual_cond_noise_aug"]
        if kwargs.get("minimax_audio_cond_noise_aug", None) is not None:
            payload["audio_cond_noise_aug"] = kwargs["minimax_audio_cond_noise_aug"]
        payload["seed"] = kwargs.get("seed", 0)
        payload["audio_scale"] = original_model.audio_scale()
        # Denoise-mask conds, exactly as native MiniMaxH3.extra_conds emits them.
        # These are what tell the DiT that masked rows are already resolved, so a
        # frozen region reads as context instead of as noise at the current
        # sigma. Replacing extra_conds without this silently dropped them for
        # every masked/inpaint-style sampling run.
        denoise_mask = kwargs.get("denoise_mask", None)
        if denoise_mask is not None and hasattr(original_model, "_denoise_mask_conds"):
            out.update(original_model._denoise_mask_conds(denoise_mask, latent_shapes))
        if cross_attn is not None and latent_shapes is not None and len(latent_shapes) > 1:
            vs = latent_shapes[0]
            # A keyframe's cond segment is sized from the TARGET's frame grid,
            # while _cond_video_rows patchifies the keyframe's own latent, so
            # the two only agree when the keyframe was encoded at the canvas
            # being sampled. Catch the disagreement here, where the sizes are
            # still meaningful, instead of letting it surface as a broadcast
            # error inside the DiT's row assignment.
            tgt_h, tgt_w = (vs[3] + 1) // 2 * 2, (vs[4] + 1) // 2 * 2
            for kf in (keyframes or ()):
                z = kf.get("latent")
                if z is None:
                    continue
                if int(z.shape[-2]) != tgt_h or int(z.shape[-1]) != tgt_w:
                    raise ValueError(
                        "MiniMax H3 Timeline: keyframe media was encoded for a "
                        f"{int(z.shape[-1])}x{int(z.shape[-2])} latent but the latent being "
                        f"sampled is {tgt_w}x{tgt_h} "
                        f"({tgt_w * 16}x{tgt_h * 16} px). A keyframe's rows are built on the "
                        "target's own grid, so the two have to match.\n"
                        "Connect the latent you are about to sample into the Conditioning "
                        "node's `latent` input -- it then defines the canvas and the "
                        "resolution/aspect_ratio widgets are ignored."
                    )
            # THE FIX for bug 2: our corrected layout builder, not the
            # native PackedLayout, so a keyframe's anchor accounts for
            # however far references push the real video origin.
            payload["layout"] = _corrected_packed_layout(
                cross_attn.shape[1], vs[2], (vs[3] + 1) // 2 * 2, (vs[4] + 1) // 2 * 2,
                latent_shapes[1][-1], keyframes=payload.get("keyframes"),
                refs=payload.get("refs"), frame_count=payload.get("frame_count"),
                audio_keyframes=payload.get("audio_keyframes"),
                pretimeline_gap=h3model.FRAME_RESCALE * 24.0 * pretimeline_gap_seconds,
                spatial_collision_offset=spatial_collision_offset,
                anchor_decouple_scale=h3model.FRAME_RESCALE * 24.0 * anchor_decouple_scale_seconds)
        out["minimax_payload"] = comfy.conds.CONDConstant(payload)
        return out

    return fixed_extra_conds


def _av_streams(latent):
    """(video, audio) out of an H3 AV latent, validated."""
    samples = latent.get("samples") if isinstance(latent, Mapping) else None
    if (samples is None or not getattr(samples, "is_nested", False)
            or len(getattr(samples, "tensors", ())) != 2):
        raise ValueError("latent must be a MiniMax H3 AV latent (nested video + audio)")
    video, audio = samples.tensors[0], samples.tensors[1]
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(
            f"latent's video stream should be [B,24,T,H,W], got {tuple(video.shape)}")
    return video, audio


def _geometry_from_latent(latent):
    """(width, height, length) of the canvas a latent actually occupies.

    A keyframe's "cond" segment is sized from the TARGET's frame grid in
    _corrected_packed_layout, while _cond_video_rows patchifies the keyframe's
    OWN latent. Those two agree only when keyframe media was encoded at the
    canvas actually being sampled -- so when a latent is supplied, that latent
    defines the canvas, rather than the resolution/aspect widgets describing
    one it may not match. The widgets can only name preset canvases anyway;
    a latent can be any valid size."""
    video, audio = _av_streams(latent)
    lh, lw = int(video.shape[3]), int(video.shape[4])
    if lh % 2 or lw % 2:
        raise ValueError(
            f"latent is {lw}x{lh} latent units; both must be EVEN because the DiT "
            "patchifies 2x2. Use a canvas whose pixel width and height are multiples of 32."
        )
    latent_t = int(video.shape[2])
    if latent_t < 2 or (latent_t - 2) % 5 != 0:
        raise ValueError(
            f"latent has latent_t={latent_t}, which is not on MiniMax H3's 5-token/"
            "17-frame grid (valid: 2, 7, 12, 17, ...) -- something resampled it along "
            "time without accounting for H3's temporal compression."
        )
    length = sum(h3model.FRAME_PER_TOKEN[k % 5] for k in range(latent_t))
    return lw * 16, lh * 16, length


def _apply_timeline_patches(model, pretimeline_gap_seconds, spatial_collision_offset,
                            anchor_decouple_scale_seconds):
    """Install the four object patches that make the timeline's semantics
    expressible, on a CLONE of `model`. Returns (patched_model, report):
    report is '' on success, or the human-readable list of missing DiT
    attributes when this ComfyUI build can't support the patches -- in which
    case NOTHING is applied and the original model comes back untouched.

    Shared by MiniMaxH3TimelineModelPatch (the standalone node) and
    MiniMaxH3ConditioningTimelineIntegration (the bundled path kept for
    existing workflows). Lifted verbatim out of the latter -- the patch
    targets, their order, and the values captured in each closure are
    unchanged."""
    incompat = _dit_compatibility_report(model)
    if incompat:
        return model, incompat

    model = model.clone()
    original_model = model.model
    model.add_object_patch(
        "extra_conds",
        _make_fixed_extra_conds(
            original_model, pretimeline_gap_seconds, spatial_collision_offset,
            anchor_decouple_scale_seconds,
        ),
    )
    # add_object_patch supports dotted paths (comfy.utils.set_attr/
    # resolve_attr), so this reaches the DiT instance nested inside
    # the model directly.
    original_diffusion_model = original_model.diffusion_model
    model.add_object_patch("diffusion_model._cond_video_rows", _make_fixed_cond_video_rows(original_diffusion_model))
    model.add_object_patch("diffusion_model._cond_audio_rows", _make_fixed_cond_audio_rows(original_diffusion_model))
    # Per-ROW apparent timestep (see _make_fixed_forward's docstring)
    # -- decouples each row's noise-injection amount (above) from how
    # "resolved" the model is told that row is, which native ties to
    # the same single scalar.
    model.add_object_patch("diffusion_model._forward", _make_fixed_forward(original_diffusion_model))
    return model, ""


_INCOMPAT_MESSAGE = (
    "[MiniMaxH3-Timeline] This ComfyUI build's MiniMax H3 model is "
    "missing {!r}, which the timeline patches rely on. "
    "Applying NOTHING (the Timeline Editor's keyframes/references "
    "still run through native MiniMax H3, minus the per-item "
    "noise_aug and the keyframe/reference origin fixes). Pin "
    "ComfyUI to a tested version or update this pack."
)


def _anchor_frame_index(item: _TimelineItem, frame_count: int) -> int | None:
    if item.anchor_seconds < 0.0:
        return None
    return max(0, min(frame_count - 1, round(item.anchor_seconds * h3.FPS)))


def _anchor_audio_frame_index(item: _TimelineItem, audio_t: int) -> int:
    """Like _anchor_frame_index but in audio-latent-frame units
    (h3.AUDIO_LATENT_FPS, not h3.FPS) -- used for an audio
    keyframe_mid's placement. Unlike the video case, unset (anchor_seconds
    < 0) has no meaningful audio equivalent of "unanchored"; callers only
    use this for keyframe_mid, which always has a real anchor_seconds."""
    return max(0, min(audio_t - 1, round(max(0.0, item.anchor_seconds) * h3.AUDIO_LATENT_FPS)))


def _combined_conditioning(clip, video_vae, audio_vae, prompt, width, height, length, ref_image_size,
                           timeline: MiniMaxH3TimelineBundle, provided_latent=None):
    items = timeline.items
    keyframe_start = next((i for i in items if i.role == KEYFRAME_START), None)
    keyframe_end = next((i for i in items if i.role == KEYFRAME_END), None)
    keyframe_mids = [i for i in items if i.role == KEYFRAME_MID]
    reference_items = [i for i in items if i.role == REFERENCE]

    if provided_latent is not None:
        # Sampling a supplied latent, so the canvas is whatever IT is --
        # width/height/length came from _geometry_from_latent. Everything below
        # encodes keyframe and reference media against that same canvas, which
        # is the requirement: a keyframe's cond rows are built on the target's
        # own grid.
        video, audio = _av_streams(provided_latent)
        latent = provided_latent
        frame_count = sum(h3model.FRAME_PER_TOKEN[k % 5] for k in range(int(video.shape[2])))
        target_audio_t = int(audio.shape[-1])
    else:
        target_audio_t = h3.temporal_shape(length)[2]
        latent, frame_count = h3._empty_av_latent(width, height, length)

    # --- keyframes ---
    # A video keyframe builds a multi-frame "cond" segment, reusing the same
    # _video_grid machinery the reference-video path uses. See
    # _corrected_packed_layout for the position-id side.
    # Audio keyframes reuse
    # _encode_reference_audio (already generic -- no length constraint like
    # video's %17==5 patchify grouping) but pins the result to a specific
    # point in the TARGET's own audio track (a new "cond_audio" segment
    # kind) instead of an independent pretimeline reference block. See
    # _corrected_packed_layout and _make_fixed_forward's seg_tag handling.
    keyframe_images = []
    keyframe_videos = []  # (frames_tensor, resolved_frame_index, noise_aug) -- multi-frame, kept separate from single-image keyframes
    audio_keyframe_sources = []  # (audio_mapping, resolved_audio_frame_index, noise_aug)
    keyframes = []

    def _load_keyframe_video(item):
        frames, _soundtrack, source_fps = _video_parts(_load_media_file(item.filename, "video"))
        # a video keyframe's audio track is deliberately dropped -- pinning
        # audio content to a specific point in the target's own audio track
        # has no architectural equivalent to reuse;
        # only the visual content is used here.
        frames = _resample_video_frames(frames, source_fps)
        frames = h3._resize(frames, width, height, "center")  # must match the target canvas exactly -- a cond segment shares the target's own frame grid, unlike a reference video's independent canvas
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        count = frames.shape[0]
        if count < 5:
            raise ValueError("Video keyframes need at least 5 frames")
        while count % 17 != 5:
            count -= 1
        return frames[:count]

    if keyframe_start is not None:
        if keyframe_start.media_type == "video":
            frames = _load_keyframe_video(keyframe_start)
            keyframe_videos.append((frames, 0, keyframe_start.noise_aug))
        elif keyframe_start.media_type == "audio":
            audio_keyframe_sources.append((_load_media_file(keyframe_start.filename, "audio"), 0, keyframe_start.noise_aug))
        elif keyframe_start.media_type == "image":
            source = _load_media_file(keyframe_start.filename, "image")[:1]
            image = h3._resize(source, width, height, "center")
            keyframe_images.append(image)
            keyframes.append({"resolved_frame_index": 0, "image": image, "noise_aug": keyframe_start.noise_aug})
        else:
            raise ValueError("keyframe_start must be an image, video, or audio")
    if keyframe_end is not None:
        if keyframe_end.media_type == "video":
            frames = _load_keyframe_video(keyframe_end)
            keyframe_videos.append((frames, frame_count - 1, keyframe_end.noise_aug))
        elif keyframe_end.media_type == "audio":
            audio_keyframe_sources.append((_load_media_file(keyframe_end.filename, "audio"), target_audio_t - 1, keyframe_end.noise_aug))
        elif keyframe_end.media_type == "image":
            source = _load_media_file(keyframe_end.filename, "image")[:1]
            image = h3._resize(source, width, height, "center")
            keyframe_images.append(image)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": image, "noise_aug": keyframe_end.noise_aug})
        else:
            raise ValueError("keyframe_end must be an image, video, or audio")
    for item in keyframe_mids:
        if item.media_type == "video":
            frames = _load_keyframe_video(item)
            keyframe_videos.append((frames, _anchor_frame_index(item, frame_count), item.noise_aug))
        elif item.media_type == "audio":
            audio_keyframe_sources.append((_load_media_file(item.filename, "audio"), _anchor_audio_frame_index(item, target_audio_t), item.noise_aug))
        elif item.media_type == "image":
            source = _load_media_file(item.filename, "image")[:1]
            image = h3._resize(source, width, height, "center")
            keyframe_images.append(image)
            keyframes.append({"resolved_frame_index": _anchor_frame_index(item, frame_count), "image": image, "noise_aug": item.noise_aug})
        else:
            raise ValueError("keyframe_mid items must be images, videos, or audio")
    for kf in keyframes:
        kf["latent"] = video_vae.encode(kf.pop("image"))
    for frames, resolved_frame_index, noise_aug in keyframe_videos:
        video_latent = video_vae.encode(frames)
        keyframes.append({
            "resolved_frame_index": resolved_frame_index, "latent": video_latent,
            "latent_t": video_latent.shape[2], "noise_aug": noise_aug,
        })
    audio_keyframes = []
    for audio_mapping, resolved_audio_frame_index, noise_aug in audio_keyframe_sources:
        audio_latent, rt = _encode_reference_audio(audio_vae, audio_mapping)
        if rt > target_audio_t:
            audio_latent = audio_latent[..., :target_audio_t]
            rt = target_audio_t
        audio_keyframes.append({
            "resolved_audio_frame_index": resolved_audio_frame_index, "latent": audio_latent,
            "latent_t": rt, "noise_aug": noise_aug,
        })

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
    if audio_keyframes:
        conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_audio_keyframes": audio_keyframes})
    if ref_blocks:
        conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
    if keyframes or audio_keyframes or ref_blocks:
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
    """Consumes a `timeline` bundle plus a CLIP and VAE wired directly from
    native loader nodes (Load CLIP with type=minimax, Load VAE -- NOT MiniMax
    H3 Easy Loader's bundle), and
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

    NO MODEL PASSES THROUGH THIS NODE. The patches that make the timeline's
    semantics work live in exactly one place -- MiniMaxH3TimelineModelPatch,
    chained next to Load Diffusion Model like MiniMaxH3SigmaShift and
    ModelAttentionBackend are. This node used to take MODEL in and hand a
    patched clone back out, which made it the only conditioning node in the
    graph shaped that way and gave two different places the model could come
    from. One way now, no ambiguity.

    video_vae is required (this node uses it to encode keyframe/reference
    media into latents); audio_vae is required only when a card is audio,
    matching native MiniMaxH3AddGuide. Neither is re-emitted as an output --
    wire VAEDecode/VAEDecodeAudio directly to the same Load VAE nodes."""

    CATEGORY = "MiniMax H3 Timeline"
    FUNCTION = "generate"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("positive", "latent", "fps")
    DESCRIPTION = "Builds combined keyframe+reference conditioning from a Timeline Editor bundle. Connect a CLIP from Load CLIP (type=minimax) and the MiniMax H3 video VAE (plus the audio VAE only if a card is audio). The MODEL does NOT come through here -- chain MiniMax H3 Timeline Model Patch next to Load Diffusion Model instead, or the per-item noise_aug and corrected layout are silently ignored."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "timeline": ("MINIMAX_H3_TIMELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (list(RESOLUTION_MEGAPIXELS.keys()), {"default": RESOLUTION_480}),
                "aspect_ratio": (list(ASPECT_RATIOS.keys()), {"default": ASPECT_WIDESCREEN}),
                "ref_image_size": (["match", "1k", "1.5k", "2k", "original"], {"default": "match"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                # When connected, THIS defines the canvas -- width/height/length
                # are read from it and the resolution/aspect widgets are ignored.
                # Leave it unconnected to build a fresh empty latent from the
                # widgets, which is the default behaviour.
                "latent": ("LATENT",),
                # Only needed when a card is AUDIO (reference audio, a reference
                # video's soundtrack, or an audio keyframe) -- same policy as
                # native MiniMaxH3AddGuide's optional audio_vae.
                "audio_vae": ("VAE",),
            },
        }

    def generate(self, clip, video_vae, timeline, prompt, resolution, aspect_ratio, ref_image_size,
                 fps=24.0, audio_vae=None, latent=None):
        if not isinstance(timeline, MiniMaxH3TimelineBundle):
            raise ValueError("Connect a MiniMax H3 Timeline Editor output")
        # No MODEL here by design. The model patches live in exactly one place --
        # MiniMaxH3TimelineModelPatch, chained next to Load Diffusion Model.
        # This node builds conditioning and nothing else, so it has the same
        # shape as every other conditioning node and there is no second way to
        # wire it. Which timeline features are in play no longer has to be
        # decided here either: fixed_extra_conds gates itself at runtime on
        # what the conditioning actually carries.
        if audio_vae is None and any(i.media_type == "audio" for i in timeline.items):
            raise ValueError(
                "This timeline has an audio card, which has to be VAE-encoded to a latent -- "
                "connect audio_vae (Load VAE with the MiniMax H3 audio VAE). "
                "It is only required when a card is audio."
            )

        # TARGET GEOMETRY. Single source of truth for the canvas everything --
        # the latent, keyframe media, reference sizing -- gets built at. A
        # connected `latent` wins over the resolution/aspect widgets: the
        # canvas has to be the one actually being sampled, and only the latent
        # knows that for certain. Nothing downstream of this point changes --
        # placement, ordering, anchoring and per-item noise_aug never learn
        # where the numbers came from.
        if latent is not None:
            width, height, length = _geometry_from_latent(latent)
        else:
            width, height = _canvas_dimensions(resolution, aspect_ratio, 0, 0)
            length = _frame_length(timeline.duration_seconds, h3.FPS)

        conditioning, latent = _combined_conditioning(
            clip, video_vae, audio_vae, prompt, width, height, length, ref_image_size, timeline,
            provided_latent=latent)

        return (conditioning, latent, float(fps))


class MiniMaxH3TimelineModelPatch:
    """Installs the timeline's model patches, standalone -- chain it next to
    Load Diffusion Model the way MiniMaxH3SigmaShift / ModelAttentionBackend
    are chained, instead of routing MODEL in and out of the Conditioning node.

    Exactly the same four patches, in the same order, with the same captured
    values (see _apply_timeline_patches). object_patches survive
    ModelPatcher.clone(), so this can sit anywhere in the model chain --
    before or after LoRA / sigma shift / attention backend.

    SAFE TO LEAVE CONNECTED for non-timeline generations: the patched
    extra_conds early-outs to native when a conditioning carries no
    minimax_keyframes / minimax_refs / minimax_audio_keyframes, so a plain
    t2v run through this model is bit-identical to an unpatched one. Without
    that early-out an unconditionally-patched model would shift every plain
    generation by pretimeline_gap.

    REQUIRED for any timeline generation. The Conditioning node has no MODEL
    input, so this is the only place the patches get installed. Omit it and
    nothing errors: an unpatched model reads a timeline payload with native's
    per-KIND seg_t and silently ignores the per-item noise_aug lists."""

    CATEGORY = "MiniMax H3 Timeline"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = ("Applies the MiniMax H3 Timeline model patches (per-item noise_aug, "
                   "corrected packed layout, per-row timesteps). Chain next to Load "
                   "Diffusion Model, then leave the Conditioning node's `model` input "
                   "unconnected. Harmless on non-timeline generations.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "pretimeline_gap_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Gap between the last reference block and the video's real "
                               "start, in seconds of rotary time."}),
                "spatial_collision_offset": ("FLOAT", {
                    "default": 64.0, "min": 0.0, "max": 512.0, "step": 1.0,
                    "tooltip": "Only used by the dormant per-reference anchoring path. "
                               "Leave as-is."}),
                "anchor_decouple_scale_seconds": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Only used by the dormant per-reference anchoring path. "
                               "Leave as-is."}),
            },
        }

    def patch(self, model, pretimeline_gap_seconds=1.0, spatial_collision_offset=64.0,
              anchor_decouple_scale_seconds=2.0):
        patched, incompat = _apply_timeline_patches(
            model, pretimeline_gap_seconds, spatial_collision_offset,
            anchor_decouple_scale_seconds)
        if incompat:
            print(_INCOMPAT_MESSAGE.format(incompat))
        else:
            print("[MiniMaxH3-Timeline] model patched: extra_conds, "
                  "diffusion_model._cond_video_rows, ._cond_audio_rows, ._forward "
                  f"(pretimeline_gap={pretimeline_gap_seconds}s)")
        return (patched,)


_UPSCALER_PACK_DIRS = ("Comfyui_Minimax_h3_latent_Upscaler",)
_upscaler_module_cache = {}


def _latent_upscaler_module():
    """Import the neural latent-upscaler backbone from the sibling custom-node
    pack, if it is installed. Cached. Returns None when it is not present --
    callers raise their own message rather than failing on an ImportError."""
    if "mod" in _upscaler_module_cache:
        return _upscaler_module_cache["mod"]
    mod = None
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pack in _UPSCALER_PACK_DIRS:
        path = os.path.join(here, pack, "nodes", "minimax_h3_latent_upscaler_3d.py")
        if not os.path.isfile(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("_mmxtl_upscaler_3d", path)
            candidate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(candidate)
            if all(hasattr(candidate, a) for a in ("load_model", "_make_norm_tensors")):
                mod = candidate
                break
        except Exception as e:  # pragma: no cover - optional dependency
            print(f"[MiniMaxH3-Timeline] latent upscaler pack found but not importable: {e}")
    _upscaler_module_cache["mod"] = mod
    return mod


def _upscale_video_latent(mod, model, z, scale, out_h, out_w, dtype, device):
    """One [B,24,T,H,W] latent through the upscaler backbone, normalized the way
    it was trained and returned on the CPU at its original dtype."""
    orig_dtype = z.dtype
    s = z.to(device=device, dtype=dtype, copy=True)
    mean, std = mod._make_norm_tensors(device, dtype)
    with torch.inference_mode():
        out = model((s - mean) / std, scale=scale,
                    target_size=(s.shape[2], out_h, out_w), enable_chunking=False)
        out = out * std + mean
    return out.to(device="cpu", dtype=orig_dtype)


class MiniMaxH3TimelineLatentUpscale:
    """Upscales an H3 AV latent AND carries its conditioning to the new canvas,
    as one node, so the two cannot end up describing different sizes.

    A keyframe's cond rows are built on the TARGET's frame grid while
    _cond_video_rows patchifies the keyframe's own latent, so a resized latent
    needs its keyframes resized with it. Emitting both from one node makes that
    structural instead of something to wire correctly: the `positive` and
    `latent` outputs are always a matched pair.

    Keyframe latents go through the same trained upscaler as the target rather
    than being interpolated -- it is the model that exists for this. References
    are left untouched: a reference carries its own canvas and is never sized
    against the target grid. Audio keyframes are untouched too; the resize is
    spatial only.

    Replaces LTXVSeparateAVLatent -> upscaler -> LTXVConcatAVLatent, and removes
    the need for a second Conditioning node at the new resolution."""

    CATEGORY = "MiniMax H3 Timeline"
    FUNCTION = "upscale"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    DESCRIPTION = ("Upscales a MiniMax H3 AV latent and brings its keyframe conditioning "
                   "with it, as a matched pair. Drop it between the two samplers of a "
                   "two-pass workflow -- no second Conditioning node, no Separate/Concat.")

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("latent_upscale_models")
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "model_name": (models if models else ["(place a model in models/latent_upscale_models)"],),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                                    "tooltip": "Spatial upscale factor. Time is unchanged."}),
            },
            "optional": {
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            },
        }

    def upscale(self, positive, latent, model_name, scale, precision="fp16"):
        mod = _latent_upscaler_module()
        if mod is None:
            raise ValueError(
                "This node needs the MiniMax H3 latent upscaler pack installed alongside "
                "it (custom_nodes/Comfyui_Minimax_h3_latent_Upscaler) -- it uses that "
                "pack's trained backbone. Install it, or upscale the latent yourself and "
                "connect the result to the Conditioning node's `latent` input instead."
            )
        if str(model_name).startswith("("):
            raise ValueError("Put a latent upscaler checkpoint in models/latent_upscale_models")

        video, audio = _av_streams(latent)
        in_h, in_w = int(video.shape[3]), int(video.shape[4])
        # Align in PIXEL space to CANVAS_MULTIPLE so the resulting latent is even
        # on both axes -- the DiT patchifies 2x2, and _geometry_from_latent
        # rejects an odd latent for the same reason.
        px_w = max(h3.CANVAS_MULTIPLE, round(in_w * 16 * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        px_h = max(h3.CANVAS_MULTIPLE, round(in_h * 16 * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        out_w, out_h = px_w // 16, px_h // 16
        if (out_w, out_h) == (in_w, in_h):
            return (positive, latent)
        if out_w < in_w or out_h < in_h:
            raise ValueError("This model only upscales (scale >= 1.0)")

        device = comfy.model_management.get_torch_device() if precision != "cpu" else torch.device("cpu")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        model = mod.load_model(model_name, device, precision)
        # the effective factor the backbone is told, averaged over both axes the
        # same way the upscaler's own size modes derive it
        eff = ((out_w / in_w) + (out_h / in_h)) / 2.0

        print(f"[MiniMaxH3-Timeline] latent {in_w}x{in_h} -> {out_w}x{out_h} "
              f"({px_w}x{px_h} px), scale={eff:.3f}")
        up_video = _upscale_video_latent(mod, model, video, eff, out_h, out_w, dtype, device)
        out_latent = {"samples": comfy.nested_tensor.NestedTensor((up_video, audio))}

        # carry the conditioning to the same canvas: keyframes only
        n_kf = 0
        new_cond = []
        for tensor, d in positive:
            nd = dict(d)
            kfs = nd.get("minimax_keyframes")
            if kfs:
                moved = []
                for kf in kfs:
                    nkf = dict(kf)          # every other key rides along untouched
                    z = kf.get("latent")
                    if z is not None and (int(z.shape[-2]) != out_h or int(z.shape[-1]) != out_w):
                        k_eff = ((out_w / int(z.shape[-1])) + (out_h / int(z.shape[-2]))) / 2.0
                        nkf["latent"] = _upscale_video_latent(
                            mod, model, z, k_eff, out_h, out_w, dtype, device)
                        nkf["latent_t"] = int(nkf["latent"].shape[2])
                        n_kf += 1
                    moved.append(nkf)
                nd["minimax_keyframes"] = moved
            new_cond.append([tensor, nd])
        if n_kf:
            print(f"[MiniMaxH3-Timeline] {n_kf} keyframe latent(s) moved to the new canvas")

        comfy.model_management.soft_empty_cache()
        return (new_cond, out_latent)


class MiniMaxH3TextEncoderLoader:
    """Loads a MiniMax H3 text encoder checkpoint the same way native "Load
    CLIP" (type=minimax) does, but exposes two things that node doesn't:

    1. config_overrides -- a JSON object threaded straight into a real
       override hook that already exists in ComfyUI core
       (model_options["qwen3vl_32b_model_config"], read in
       comfy/sd1_clip.py's SDClipModel.__init__ and merged into the config
       dataclass the text-encoder model is built from -- see
       comfy/text_encoders/llama.py's Qwen3VL_32BConfig). This isn't a new
       mechanism; it's exposing one that already exists in core but that no
       stock node UI surfaces, so it was previously reachable only by
       editing comfy/text_encoders/llama.py directly.

    2. A pre-flight check that reads the checkpoint's own tensor shapes
       (embed_tokens.weight for vocab_size/hidden_size, the highest
       "model.layers.N." index present for layer count, layer 0's
       mlp.gate_proj.weight for intermediate_size) and compares them
       against ComfyUI's one hardcoded MiniMax H3 default
       (Qwen3VL_32BConfig: hidden_size=5120, num_hidden_layers=50,
       intermediate_size=25600 -- ComfyUI's own comment on that class notes
       this is "truncated to the first 50 of 64 layers"). If the checkpoint
       being loaded doesn't match that shape and no override was given,
       this raises an error naming exactly which fields differ and the
       JSON to paste into config_overrides -- instead of either silently
       loading only the first 50 layers of a larger checkpoint (no error at
       all, wrong output) or crashing deep inside a weight-copy with a
       shape-mismatch message that names a tensor, not a fix.

    This does not make an architecturally different text encoder work --
    only Qwen3-VL-32B-based checkpoints shaped differently than ComfyUI's
    one hardcoded default (e.g. MiniMax's own un-truncated release, or a
    fine-tune with a different layer count/hidden size)."""

    CATEGORY = "MiniMax H3 Timeline"
    FUNCTION = "load"
    RETURN_TYPES = ("CLIP",)
    DESCRIPTION = ("Loads a MiniMax H3 (Qwen3-VL-32B) text encoder checkpoint with an "
                   "optional JSON config_overrides for checkpoints shaped differently than "
                   "ComfyUI's built-in default (e.g. an un-truncated official release, or a "
                   "fine-tune with a different layer count/hidden size). Detects the real "
                   "shape from the checkpoint's own tensors and tells you what to override "
                   "if it doesn't match what would otherwise be silently assumed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
            },
            "optional": {
                "config_overrides": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "JSON object of Qwen3VL_32BConfig field overrides, e.g. "
                               '{"num_hidden_layers": 64, "hidden_size": 5120}. Leave empty '
                               "to use ComfyUI's built-in default (50-layer truncated config). "
                               "Only used if the pre-flight check finds/needs an override -- "
                               "if it detects a mismatch with this left empty, the error message "
                               "gives you the exact JSON to paste here.",
                }),
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    @staticmethod
    def _detected_shape(sd):
        """Reads architecture-relevant dims straight out of the checkpoint's
        own tensors. Returns a dict of {config_field: detected_value},
        omitting fields whose defining tensor isn't present."""
        detected = {}

        layer_idxs = [int(m.group(1)) for key in sd
                      for m in [re.match(r"^model\.layers\.(\d+)\.", key)] if m]
        if layer_idxs:
            detected["num_hidden_layers"] = max(layer_idxs) + 1

        embed = sd.get("model.embed_tokens.weight")
        if embed is not None:
            detected["vocab_size"] = embed.shape[0]
            detected["hidden_size"] = embed.shape[1]

        gate = sd.get("model.layers.0.mlp.gate_proj.weight")
        if gate is not None:
            detected["intermediate_size"] = gate.shape[0]

        return detected

    def load(self, clip_name, config_overrides="", device="default"):
        overrides = {}
        if config_overrides and config_overrides.strip():
            try:
                overrides = json.loads(config_overrides)
            except json.JSONDecodeError as e:
                raise ValueError(f"config_overrides is not valid JSON: {e}") from e
            if not isinstance(overrides, dict):
                raise ValueError('config_overrides must be a JSON object, e.g. {"num_hidden_layers": 64}')

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        sd, _metadata = comfy.utils.load_torch_file(clip_path, safe_load=True, return_metadata=True)

        # This shape check only applies to the MiniMax H3 architecture --
        # detect via the same key comfy.sd.detect_te_model() checks for it,
        # so an unrelated checkpoint just proceeds to native loading (and
        # native loading's own, clearer error if it's the wrong type entirely).
        is_minimax_arch = ("visual.deepstack_merger_list.0.norm.weight" in sd
                            and "model.layers.49.self_attn.q_proj.weight" in sd)
        if is_minimax_arch:
            detected = self._detected_shape(sd)
            defaults = {"num_hidden_layers": 50, "hidden_size": 5120,
                        "vocab_size": 151936, "intermediate_size": 25600}
            mismatches = {}
            for field, det_val in detected.items():
                effective = overrides.get(field, defaults[field])
                if det_val != effective:
                    mismatches[field] = det_val
            if mismatches:
                raise ValueError(
                    "This checkpoint's actual shape doesn't match the config it would load "
                    f"with. Detected directly from its own tensors: {mismatches}. "
                    "ComfyUI's built-in MiniMax H3 default assumes "
                    f"{ {k: defaults[k] for k in mismatches} } (a checkpoint truncated to 50 "
                    "of 64 layers -- see comfy/text_encoders/llama.py's Qwen3VL_32BConfig). "
                    "Paste this into config_overrides to load it as its actual shape instead: "
                    + json.dumps(mismatches)
                )

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        if overrides:
            model_options["qwen3vl_32b_model_config"] = overrides

        clip = comfy.sd.load_text_encoder_state_dicts(
            [sd], embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.MINIMAX, model_options=model_options,
        )
        return (clip,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TimelineEditor": MiniMaxH3TimelineEditor,
    "MiniMaxH3ConditioningTimelineIntegration": MiniMaxH3ConditioningTimelineIntegration,
    "MiniMaxH3TimelineModelPatch": MiniMaxH3TimelineModelPatch,
    "MiniMaxH3TimelineLatentUpscale": MiniMaxH3TimelineLatentUpscale,
    "MiniMaxH3TextEncoderLoader": MiniMaxH3TextEncoderLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TimelineEditor": "MiniMax H3 Timeline Editor",
    "MiniMaxH3ConditioningTimelineIntegration": "MiniMax H3 Conditioning (Timeline Integration)",
    "MiniMaxH3TimelineModelPatch": "MiniMax H3 Timeline Model Patch",
    "MiniMaxH3TimelineLatentUpscale": "MiniMax H3 Timeline Latent Upscale",
    "MiniMaxH3TextEncoderLoader": "MiniMax H3 Text Encoder Loader (config override)",
}
