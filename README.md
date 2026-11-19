# platonic-touch

**Do touch encoders converge with vision and language?**

The [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) (Huh et al., 2024)
argues that as models get better, their internal representations of the world become more
alike, even across modalities. The paper measured vision and language only. This project
asks the same question for **touch**, using open tactile encoders (Meta's
[Sparsh](https://github.com/facebookresearch/sparsh)) and the
[TVL](https://tactile-vlm.github.io/) touch-vision-language dataset, all on a 16GB Apple
Silicon laptop.

It is also a learning project: each stage pairs one ML skill with one MLOps/SRE skill.

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| 0 | Clean workbench: uv-locked env, PyTorch on MPS + MLX, tests, CI | in progress |
| 1 | Reproduce the paper's vision-language alignment trend (mutual k-NN) | planned |
| 2 | Add touch: touch-vision and touch-language alignment | planned |
| 3 | Train touch into a shared space (projection head, contrastive fine-tune) | planned |
| 4 | A tiny touch "world model" predicting the next touch embedding | planned |
| 5 | Scale out on rented GPUs and contribute upstream | planned |

## Quickstart

Requires an Apple Silicon Mac (or Linux for CPU-only tests) and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync                               # create .venv from uv.lock
uv run python scripts/check_env.py    # verify MLX + PyTorch MPS
uv run pytest                         # run tests
```

Or use the Makefile: `make setup`, `make check`, `make test`, `make lint`.

## Layout

```
src/platonic_touch/   reusable library code (device selection, metrics, data loaders)
scripts/              one-off entry points (env check, downloads, embedding extraction)
experiments/          one folder per roadmap stage: configs, notes and results
configs/              shared experiment configs
notebooks/            exploration only; anything reusable moves to src/
tests/                fast tests that run in CI without a GPU
docs/                 write-ups and the papercuts log
data/                 datasets and cached embeddings (gitignored)
```

## License

MIT, see [LICENSE](LICENSE).
