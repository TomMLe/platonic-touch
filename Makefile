.PHONY: setup check test lint format

setup:  ## create/refresh .venv from uv.lock
	uv sync

check:  ## verify MLX + PyTorch MPS on this machine
	uv run python scripts/check_env.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
