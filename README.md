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

Every card also has a **`noise_aug`** field (default 0.999, 1.0 = used
exactly as given). This is a real native MiniMax H3 mechanism made of two
parts that are normally tied to the same one global scalar: how much noise
gets blended into that row's own content, and how "resolved" the model is
told that row is (its apparent denoising timestep, which affects how much
the AdaLN/RoPE embeddings let it be reasoned about relative to the rest of
the sequence). Native code reads exactly one such value per generation for
each of those two things and applies it identically to every row of a kind;
this project patches `_cond_video_rows`/`_cond_audio_rows` (content) and
`_forward`'s `seg_t` construction (resolved-ness), via the same
`model.clone().add_object_patch(...)` mechanism used for the bug fixes
above, so each row's own value drives both independently -- verified
directly, not assumed (two rows with identical source content but
different `noise_aug` produced different, independently-controlled output).

Testing this against an actual hard-cut transition (see Tips below) found
`noise_aug` isn't a fix for that on its own -- lowering it only trades
cut-sharpness for content fidelity (noisier/less-accurate reproduction of
that item) along one axis, at every point along the range. The real cause
turned out to be something else entirely.

The Timeline Editor also has global `visual_cond_noise_aug` /
`audio_cond_noise_aug` settings, used as the fallback for any row that
doesn't set its own (e.g. if per-item support is ever removed, or for
items created before this existed).

An earlier iteration also
exposed `pretimeline_gap_seconds` (how far a reference's pre-timeline slot
sits from the video's real start); a direct A/B test (0.3s vs 3.0s, same
seed/prompt/references otherwise) showed no meaningfully different result,
so it was removed from the UI for the same reason the anchoring system was
-- it's a fixed default internally, not something worth exposing as a
setting users could mistakenly change expecting an effect.

### MiniMax H3 Conditioning (Timeline Integration)

Takes the timeline plus a `MODEL` / `CLIP` / `VAE` / `VAE` wired **directly
from native ComfyUI loader nodes** -- `Load Diffusion Model`, `Load CLIP`
with `type` set to `minimax`, and two `Load VAE` nodes -- not a bundle node
with internal fallback logic. Whatever checkpoint is visibly connected on
the canvas is what gets used; there's nothing to silently substitute.

`video_vae`/`audio_vae` are required inputs -- this node uses them
internally to encode keyframe/reference media into latents -- but are not
re-emitted as outputs. Outputs are just `model`, `positive` (conditioning),
`latent`, `fps`. Wire the *same* `Load VAE` nodes connected here directly to
`VAEDecode`/`VAEDecodeAudio` as well (one `Load VAE` feeding both this node
and the decoder, not a chain through this node) -- fan-out from a single
loader, not a passthrough dependency.

**`model` is the one output you must actually route through this node --
never bypass it with the loader's own `MODEL` output downstream (LoRA,
attention patches, sampling, etc.).** They are not the same object when a
keyframe and a reference are combined or a mid-clip keyframe is used: this
node clones the model and attaches the corrected `extra_conds` here --
that's where the two bug fixes above actually get applied. Wiring
`Load Diffusion Model`'s output around this node instead silently
reintroduces both bugs, with no error.

Typing `@` in the prompt shows the timeline's actual reference items and
their real `<Picture N>` / `<Video N>` / `<Audio N>` tags, computed the same
way the backend resolves them at generation time -- no guessing which tag
maps to which uploaded item.

## Tips

Patterns found through actual testing, not assumed -- added to as more come up.

- **A mid-clip keyframe "hard cutting" into place instead of transitioning
  smoothly is a duration problem, not a noise_aug problem.** A `Mid`
  keyframe is pinned to an exact frame; if `duration_seconds` is short,
  there simply aren't enough frames between the clip's start and that
  anchor point for the model to interpolate through, so it jumps instead of
  easing in. `noise_aug` only changes how that keyframe's own row is
  represented -- it can't create temporal room that isn't there. Increasing
  `duration_seconds` fixed a hard cut that no `noise_aug` value did.

## Requirements

- ComfyUI with MiniMax H3 support (`comfy_extras/nodes_minimax_h3.py`).
- A MiniMax H3 diffusion checkpoint, a `minimax`-type CLIP/text-encoder
  checkpoint, and the MiniMax H3 video + audio VAEs, loadable through
  ComfyUI's native `Load Diffusion Model` / `Load CLIP` / `Load VAE` nodes.

No extra Python dependencies beyond what ComfyUI itself already ships with.

## License

MIT -- see [LICENSE](LICENSE).
