# ComfyUI-MiniMaxH3-Easy

Combined keyframe + reference conditioning for MiniMax H3 in ComfyUI: one
timeline where each item is either a keyframe (start / end / mid-clip frame)
or a reference (identity/character conditioning), generated together in a
single pass. Two nodes: `MiniMax H3 Timeline Editor` and `MiniMax H3
Conditioning (Timeline Integration)`.

## Why this exists

Native ComfyUI has two real bugs that prevent keyframes and references from
being combined in one MiniMax H3 generation:

1. `comfy/model_base.py`'s `MiniMaxH3.extra_conds` builds the conditioning
   video-latent list from keyframes, then unconditionally **overwrites** it
   with the reference list instead of concatenating the two -- with both
   present, the keyframe's own image data never reaches the model at all.
2. `comfy/ldm/minimax/model.py`'s `PackedLayout` anchors a "first" keyframe's
   rotary position to the raw text length, which is only correct when
   nothing precedes the video segment. Each reference pushes the video's
   real start later in the sequence, so with references present the
   keyframe ends up anchored to a point *before* the video actually begins.

Both were confirmed directly against the installed ComfyUI source, not
assumed from community explanations. This project patches the loaded
model's `extra_conds` (via `model.clone().add_object_patch(...)`, the
standard non-invasive override mechanism -- no core files are edited) with a
corrected version that concatenates both latent lists and anchors keyframes
against the real, post-reference video origin.

Multiple simultaneous reference characters work correctly out of the box.
An earlier iteration of this project added a whole per-reference
position/strength ("anchoring") system to try to force multiple references
into one shared scene; extensive same-seed A/B testing showed that plain,
unanchored multi-reference conditioning already produces clean co-presence
once the two bugs above are fixed, so that system was removed rather than
kept as an unused, confusing knob.

## Nodes

### MiniMax H3 Timeline Editor

Upload media directly as cards (the same `/upload/image` endpoint native
`LoadImage` uses) rather than wiring in separate loader nodes -- click a card
to upload an image, video, or audio file, then mark its role:

- **Start** / **End** -- pins that image to the clip's first/last frame.
- **Mid** -- a keyframe placed at any point in the clip (`at:` seconds, not
  restricted to the two endpoints).
- **Ref** -- a reference for identity/character conditioning.

Global settings: `duration_seconds` and `pretimeline_gap_seconds` (how far a
reference's default pre-timeline slot sits from the video's real start).

### MiniMax H3 Conditioning (Timeline Integration)

Takes the timeline plus a `MODEL` / `CLIP` / `VAE` / `VAE` wired **directly
from native ComfyUI loader nodes** -- `Load Diffusion Model`, `Load CLIP`
with `type` set to `minimax`, and two `Load VAE` nodes -- not a bundle node
with internal fallback logic. Whatever checkpoint is visibly connected on
the canvas is what gets used; there's nothing to silently substitute.

Outputs `model`, `positive` (conditioning), `latent`, `video_vae`,
`audio_vae`, `fps` -- wire these into a normal `SamplerCustomAdvanced` /
`VAEDecode` / `VAEDecodeAudio` / `CreateVideo` chain like any other ComfyUI
video workflow.

Typing `@` in the prompt shows the timeline's actual reference items and
their real `<Picture N>` / `<Video N>` / `<Audio N>` tags, computed the same
way the backend resolves them at generation time -- no guessing which tag
maps to which uploaded item.

## Requirements

- ComfyUI with MiniMax H3 support (`comfy_extras/nodes_minimax_h3.py`).
- A MiniMax H3 diffusion checkpoint, a `minimax`-type CLIP/text-encoder
  checkpoint, and the MiniMax H3 video + audio VAEs, loadable through
  ComfyUI's native `Load Diffusion Model` / `Load CLIP` / `Load VAE` nodes.

No extra Python dependencies beyond what ComfyUI itself already ships with.

## License

MIT -- see [LICENSE](LICENSE).
