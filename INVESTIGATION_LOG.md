# Investigation log

Records of investigations into this project, kept separate from the README
because they're not documentation of how to use the nodes -- they're a
record of what was tried, what turned out to be true, and what turned out
to be a dead end, so none of it has to be re-discovered later.

## "Music playing before it should" with an audio keyframe

Not a fix, a record of a dead end.

**Symptom**: pinning music as an audio keyframe at second 4 of a clip (with
a prompt explicitly describing silence before that point) reliably played
*some* music before second 4 too, not the pinned track specifically --
unrelated content, not a copy of it.

**What was verified true**, in order:

1. Attention in this model is completely unmasked everywhere (`mask=None`
   in every call, confirmed by reading `comfy/ldm/minimax/model.py`) --
   every token can attend to every other token, every layer, with no
   temporal locality at all. RoPE only biases the attention *score* by
   relative distance; it never blocks attention outright.
2. Given that, a pinned audio keyframe's content is visible to (and can
   influence) every row of the target audio, not just nearby ones --
   confirmed by testing that the effect wasn't limited to *before* the
   anchor; it showed up throughout the clip.
3. A real per-keyframe containment mechanism was built and verified working
   two different ways:
   - First as an actual attention mask blocking (target-audio row, pinned
     keyframe row) pairs. Confirmed correct via direct position-math tests,
     then confirmed genuinely too slow to use: any attention mask at all
     forces PyTorch/the comfy-kitchen backend off its fused kernel path
     onto a much slower fallback that materializes the full seq_len^2
     score matrix -- benchmarked at this project's real sequence length
     (~24k tokens): 63ms unmasked vs 413ms masked per call, on one of 50
     blocks, times 8 sampling steps.
   - Rebuilt without ever passing a mask at all: split each block's
     attention into two plain unmasked calls using K/V *slicing* instead
     -- one call for "privileged" rows (the keyframe + the exact target
     rows at its own pinned time, unrestricted), one for everyone else
     (K/V with the keyframe's rows physically excluded). Benchmarked close
     to unmasked baseline (175.6ms vs 159.3ms) instead of the masked
     413ms -- genuinely fast, and confirmed live: roughly 4.7x faster
     generation than the masked version at real scale.
4. With that containment mechanism live and verified working (the pinned
   track played recognizably and only at its anchor point), the "music
   before it should play" symptom *did not go away*. Since containment
   makes it architecturally impossible for pre-anchor generation to have
   any attention path to the pinned content at all (direct or indirect,
   at every layer), this ruled out attention leakage as the cause,
   definitively, rather than confirming it.
5. Tested with **no audio keyframe present at all**, same prompt: the
   model still added music before the point the prompt says music should
   start. That's the real finding -- the entire premise of the
   investigation (pinned-content leakage) was wrong. The behavior is
   coming from the prompt/model's own tendency to add background music,
   with the audio-keyframe feature never involved in causing it at all.

**Net result**: the containment mechanism was reverted (it also introduced
its own hard-cut-style artifact at the release point, on top of not fixing
anything real). Both versions (mask-based and K/V-slicing) exist in git
history if per-keyframe containment is ever needed for an actually
attention-caused problem in the future. If you're fighting music appearing
where it shouldn't, test with the audio keyframe removed entirely before
assuming it's the cause -- it likely isn't.
