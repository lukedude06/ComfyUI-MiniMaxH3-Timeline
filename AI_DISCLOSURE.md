# AI Disclosure

Most of the code in this repository was written by Claude (Anthropic),
via Claude Code, working with the repo owner over a sustained,
multi-session process -- not a single one-shot generation. That process
included: reading the installed MiniMax H3 / ComfyUI source directly to
find the two native bugs this project fixes, building and empirically
testing each feature (per-item `noise_aug`, video keyframes, audio
keyframes), and multiple real dead ends along the way that were tried,
benchmarked or generation-tested, and reverted when they didn't hold up
-- documented in git history and in the README's Tips section rather than
edited out.

The repo owner directed the work throughout: deciding what to build,
running the actual test generations, catching several things the model
got wrong or over-claimed along the way, and making the final call on
every design decision documented in this README.

This project doesn't make a legal claim about authorship one way or the
other -- copyright treatment of AI-assisted code isn't settled law at the
time of writing. What's stated above is a factual account of how the code
was actually produced, offered so anyone using or reading this repo has
that context.
