# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## What this is
A learning + research project: measure whether touch encoders align with vision and language
encoders (Platonic Representation Hypothesis), on a 16GB Apple Silicon Mac (M6).
The owner is an SRE learning ML. Explain ML concepts briefly when introducing them.

## Commands
- Install / sync env: `uv sync` (never `pip install`; add deps with `uv add <pkg>`, dev deps with `uv add --dev <pkg>`)
- Env check: `uv run python scripts/check_env.py`
- Tests: `uv run pytest`
- Lint + format: `uv run ruff check . && uv run ruff format .`

## Conventions
- PyTorch is the main framework; MLX is used for comparison and ports.
- Device-agnostic code only: use `platonic_touch.device.get_device()`, never hardcode `"mps"` or `"cuda"`.
- Call `seed_everything()` at the start of every experiment.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if an op is missing on MPS, and log it in `docs/papercuts.md`.
- Never commit data, model weights or cached embeddings; they live in `data/` (gitignored).
- Tests in `tests/` must run on CPU in seconds (CI is Linux, no GPU).
- Library code goes in `src/platonic_touch/`; experiments in `experiments/stageN_*/`.

## Memory budget
16GB unified memory shared by CPU, GPU and macOS. Plan on ~10GB for ML. Prefer frozen
encoders + cached embeddings; use bfloat16 and gradient accumulation before bigger batches.

## Working style
- Start non-trivial features in plan mode; keep diffs small.
- "Explain it back": the owner must be able to explain any code before it is merged.
- Record friction (broken installs, missing MPS ops, unclear docs) in `docs/papercuts.md`;
  it is the backlog for open source contributions.
