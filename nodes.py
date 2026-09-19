"""MiniMax H3 Timeline Editor + combined keyframe/reference conditioning.

Nodes:
  MiniMaxH3TimelineEditor                   cards -> a timeline bundle
  MiniMaxH3ConditioningTimelineIntegration  bundle -> conditioning + latent + fps
  MiniMaxH3TextEncoderLoader                Load CLIP with a config override
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
import comfy.sd
import comfy.utils
from comfy_api.latest import InputImpl
from comfy_extras.nodes_audio import load as _load_audio_waveform

# Timeline conditioning now targets ComfyUI's public MiniMax H3 payload
# directly. FL2VA and Ref2VA use the same model class; the checkpoint weights,
# not a custom packed-layout fork, provide their different tuning.
import comfy.ldm.minimax.model as h3model
from comfy_extras import nodes_minimax_h3 as h3

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


@dataclass(frozen=True)
class MiniMaxH3TimelineBundle:
    items: tuple[_TimelineItem, ...]
    duration_seconds: float
    # Native MiniMax H3 modality-wide conditioning values. These remain global:
    # every visual conditioning row uses the visual value and every audio row
    # uses the audio value, matching core ComfyUI.
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
                # Native per-modality values shared by all conditioning rows.
                # 0.999 / 1.0 are ComfyUI's MiniMax H3 defaults.
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
            # Deliberately ignore legacy card-level noise_aug values. Noise
            # augmentation is modality-wide, matching native MiniMax H3.
            items.append(_TimelineItem(media_type, filename, role, index, anchor_seconds))
        keyframe_starts = [i for i in items if i.role == KEYFRAME_START]
        keyframe_ends = [i for i in items if i.role == KEYFRAME_END]
        if len(keyframe_starts) > 1 or len(keyframe_ends) > 1:
            raise ValueError("MiniMax H3 Timeline Editor accepts at most one keyframe_start and one keyframe_end item")
        return (MiniMaxH3TimelineBundle(
            tuple(items), float(duration_seconds),
            visual_cond_noise_aug=float(visual_cond_noise_aug),
            audio_cond_noise_aug=float(audio_cond_noise_aug),
        ),)


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

    Native H3 keyframe rows use the target's frame grid, so when a latent is
    supplied it defines the canvas rather than resolution/aspect widgets that
    may not describe its actual size."""
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


def _native_combined_conditioning(clip, video_vae, audio_vae, prompt, width, height, length,
                                  ref_image_size, timeline: MiniMaxH3TimelineBundle,
                                  provided_latent=None):
    """Compile the editor's unified timeline to ComfyUI's native H3 contract.

    Every card participates in one Qwen multimodal presentation, independent
    of role. The role only decides whether its encoded DiT payload is placed
    in ``minimax_keyframes`` or ``minimax_refs``. This is the important shared
    FL2VA/Ref2VA abstraction: changing a card from Ref to Start/Mid/End does
    not switch pipelines or text encoders.
    """
    if provided_latent is not None:
        video, audio = _av_streams(provided_latent)
        latent = provided_latent
        frame_count = sum(h3model.FRAME_PER_TOKEN[k % 5] for k in range(int(video.shape[2])))
        target_audio_t = int(audio.shape[-1])
    else:
        latent, frame_count = h3._empty_av_latent(width, height, length)
        _video, audio = _av_streams(latent)
        target_audio_t = int(audio.shape[-1])

    presentation_items: list[dict] = []
    keyframes: list[dict] = []
    ref_blocks: list[dict] = []
    tag_by_input: dict[int, str] = {}
    soundtrack_pairs: list[tuple[int, int]] = []
    cards_by_type = {
        media_type: [item for item in timeline.items if item.media_type == media_type]
        for media_type in ("image", "video", "audio")
    }

    def role_frame_index(item: _TimelineItem, guide_frames=1) -> int:
        if item.role == KEYFRAME_START:
            index = 0
        elif item.role == KEYFRAME_END:
            index = frame_count - guide_frames
        elif item.role == KEYFRAME_MID:
            index = round(item.anchor_seconds * h3.FPS)
        else:
            raise ValueError("Reference cards do not have a keyframe index")
        if index < 0 or index + guide_frames > frame_count:
            label = item.filename.rsplit("/", 1)[-1]
            raise ValueError(
                f"{label}: a {guide_frames}-frame guide at frame {index} does not fit "
                f"inside the target's {frame_count} frames"
            )
        return index

    def reference_image(image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        image_h, image_w = int(image.shape[1]), int(image.shape[2])
        size_mode = str(ref_image_size or REF_IMAGE_MATCH)
        if size_mode == REF_IMAGE_ORIGINAL:
            target_w, target_h = _original_reference_size(image_w, image_h)
            if target_w == image_w and target_h == image_h:
                return image[:1], target_w, target_h
            if image_w >= h3.CANVAS_MULTIPLE and image_h >= h3.CANVAS_MULTIPLE:
                top = (image_h - target_h) // 2
                left = (image_w - target_w) // 2
                return image[:1, top:top + target_h, left:left + target_w, :], target_w, target_h
            return h3._resize(image[:1], target_w, target_h, "disabled"), target_w, target_h
        target_area = width * height if size_mode == REF_IMAGE_MATCH else REFERENCE_IMAGE_AREAS.get(
            size_mode, REFERENCE_IMAGE_AREAS[REF_IMAGE_1K])
        scale = min(1.0, math.sqrt(target_area / max(1, image_w * image_h)))
        target_w, target_h = _reference_aligned_size(image_w, image_h, scale)
        return h3._resize(image[:1], target_w, target_h, "disabled"), target_w, target_h

    # Bucket only by media type, never by role. This keeps <Picture>/<Video>/
    # <Audio> ordinals stable when a card is toggled between Ref and a
    # keyframe role.
    for picture_ordinal, item in enumerate(cards_by_type["image"], start=1):
        source = _load_media_file(item.filename, "image")
        if item.role == REFERENCE:
            image, target_w, target_h = reference_image(source)
            z = video_vae.encode(image)
            ref_blocks.append({
                "kind": "image", "latent_h": int(z.shape[-2]),
                "latent_w": int(z.shape[-1]), "latent": z,
            })
        else:
            image = h3._resize(source[:1], width, height, "center")
            keyframes.append({
                "resolved_frame_index": role_frame_index(item),
                "latent": video_vae.encode(image),
            })
        presentation_items.append({"type": "image", "data": image})
        tag_by_input[item.item_index] = f"<Picture {picture_ordinal}>"

    audio_ordinal = 0
    for video_ordinal, item in enumerate(cards_by_type["video"], start=1):
        frames, soundtrack, source_fps = _video_parts(_load_media_file(item.filename, "video"))
        frames = _resample_video_frames(frames, source_fps)
        if item.role == REFERENCE:
            video_h, video_w = int(frames.shape[1]), int(frames.shape[2])
            canvas_w, canvas_h = h3.adapt_canvas(video_w, video_h)
            if video_w * video_h < canvas_w * canvas_h:
                canvas_w = max(h3.CANVAS_MULTIPLE, round(video_w / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
                canvas_h = max(h3.CANVAS_MULTIPLE, round(video_h / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
            frames = h3._resize(frames, canvas_w, canvas_h, "disabled")
        else:
            canvas_w, canvas_h = width, height
            frames = h3._resize(frames, width, height, "center")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        guide_frames = int(frames.shape[0])
        if guide_frames < 5:
            raise ValueError(f"{item.filename}: MiniMax H3 video cards need at least 5 frames")
        while guide_frames % 17 != 5:
            guide_frames -= 1
        frames = frames[:guide_frames]
        video_latent = video_vae.encode(frames)

        audio_latent = None
        audio_t = 0
        if soundtrack is not None:
            if audio_vae is None:
                raise ValueError(f"{item.filename}: its soundtrack requires the MiniMax H3 audio VAE")
            audio_latent, audio_t = _encode_reference_audio(audio_vae, soundtrack)
            audio_ordinal += 1
            soundtrack_pairs.append((audio_ordinal, video_ordinal))
            presentation_items.append({"type": "audio"})

        if item.role == REFERENCE:
            ref_blocks.append({
                "kind": "video_audio" if audio_t else "video",
                "latent_t": int(video_latent.shape[2]),
                "latent_h": int(video_latent.shape[-2]),
                "latent_w": int(video_latent.shape[-1]),
                "ref_audio_t": audio_t,
                "latent": video_latent,
                "audio_latent": audio_latent,
            })
        else:
            keyframe = {
                "resolved_frame_index": role_frame_index(item, guide_frames),
                "latent": video_latent,
            }
            if audio_latent is not None:
                max_rt = math.floor(target_audio_t - h3model.FRAME_RESCALE * keyframe["resolved_frame_index"])
                if max_rt < 1:
                    raise ValueError(f"{item.filename}: its soundtrack starts past the target audio track")
                keyframe["audio_latent"] = audio_latent[..., :max_rt].clone() if audio_t > max_rt else audio_latent
            keyframes.append(keyframe)

        sample_indexes = list(range(0, guide_frames, h3.FPS // 2))
        presentation_items.append({
            "type": "video", "data": frames[sample_indexes],
            "timestamps": [i / 2.0 for i in range(len(sample_indexes))],
        })
        tag_by_input[item.item_index] = f"<Video {video_ordinal}>"

    for item in cards_by_type["audio"]:
        if audio_vae is None:
            raise ValueError(f"{item.filename}: audio cards require the MiniMax H3 audio VAE")
        audio_latent, audio_t = _encode_reference_audio(audio_vae, _load_media_file(item.filename, "audio"))
        audio_ordinal += 1
        presentation_items.append({"type": "audio"})
        tag_by_input[item.item_index] = f"<Audio {audio_ordinal}>"
        if item.role == REFERENCE:
            ref_blocks.append({"kind": "audio", "ref_audio_t": audio_t, "audio_latent": audio_latent})
        else:
            if item.role == KEYFRAME_END:
                frame_index = max(0, math.floor((target_audio_t - audio_t) / h3model.FRAME_RESCALE))
            else:
                frame_index = role_frame_index(item)
            max_rt = math.floor(target_audio_t - h3model.FRAME_RESCALE * frame_index)
            if max_rt < 1:
                raise ValueError(f"{item.filename}: its keyframe starts past the target audio track")
            keyframes.append({
                "resolved_frame_index": frame_index,
                "audio_latent": audio_latent[..., :max_rt].clone() if audio_t > max_rt else audio_latent,
            })

    resolved_prompt = _resolve_reference_prompt(
        prompt, tag_by_input, soundtrack_pairs, len(cards_by_type["video"]), len(cards_by_type["audio"]))
    tokens = clip.tokenize(resolved_prompt, minimax_ref_items=presentation_items)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    values = {
        "minimax_visual_cond_noise_aug": timeline.visual_cond_noise_aug,
        "minimax_audio_cond_noise_aug": timeline.audio_cond_noise_aug,
    }
    if keyframes:
        values["minimax_keyframes"] = keyframes
    if ref_blocks:
        values["minimax_refs"] = ref_blocks
    if keyframes or ref_blocks:
        conditioning = node_helpers.conditioning_set_values(conditioning, values)
    return conditioning, latent


class MiniMaxH3ConditioningTimelineIntegration:
    """Consumes a `timeline` bundle plus a CLIP and VAE wired directly from
    native loader nodes (Load CLIP with type=minimax, Load VAE -- NOT MiniMax
    H3 Easy Loader's bundle), and
    builds one native conditioning object carrying keyframes and references
    together. FL2VA and Ref2VA checkpoints use this same conditioning schema;
    a card's role chooses its payload bucket, not a different pipeline.

    Deliberately takes raw native-loader connections instead of a bundle:
    MiniMaxH3Bundle.model_for() silently substitutes whichever H3 checkpoint
    IS configured when the one a mode prefers is left unset ("None"), so
    which checkpoint a generation actually runs on can differ from what a
    dropdown shows, and only becomes visible after the fact (in the console
    log). Wiring a specific Load Diffusion Model node's output directly
    means there is nothing to substitute -- what's connected on the canvas,
    before you run anything, IS what gets used.

    No MODEL passes through this node. ComfyUI's MiniMaxH3 model already
    combines ``minimax_keyframes`` and ``minimax_refs`` and builds the packed
    layout, so the diffusion model can be connected directly to the sampler.

    video_vae is required (this node uses it to encode keyframe/reference
    media into latents); audio_vae is required only when a card is audio,
    matching native MiniMaxH3AddGuide. Neither is re-emitted as an output --
    wire VAEDecode/VAEDecodeAudio directly to the same Load VAE nodes."""

    CATEGORY = "MiniMax H3 Timeline"
    FUNCTION = "generate"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("positive", "latent", "fps")
    DESCRIPTION = "Builds native MiniMax H3 keyframe+reference conditioning from one Timeline Editor bundle. Connect the diffusion model directly to your normal sampler/model chain; no Timeline model patch is required."

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
        # This node compiles conditioning only. The connected checkpoint flows
        # through the user's normal model chain without any timeline patch.
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
        # placement, ordering and anchoring never learn
        # where the numbers came from.
        if latent is not None:
            width, height, length = _geometry_from_latent(latent)
        else:
            width, height = _canvas_dimensions(resolution, aspect_ratio, 0, 0)
            length = _frame_length(timeline.duration_seconds, h3.FPS)

        conditioning, latent = _native_combined_conditioning(
            clip, video_vae, audio_vae, prompt, width, height, length, ref_image_size, timeline,
            provided_latent=latent)

        return (conditioning, latent, float(fps))


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
    "MiniMaxH3TextEncoderLoader": MiniMaxH3TextEncoderLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TimelineEditor": "MiniMax H3 Timeline Editor",
    "MiniMaxH3ConditioningTimelineIntegration": "MiniMax H3 Conditioning (Timeline Integration)",
    "MiniMaxH3TextEncoderLoader": "MiniMax H3 Text Encoder Loader (config override)",
}
