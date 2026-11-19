"""Pick the best available PyTorch device so the same code runs on a Mac, a CUDA box, or CI.

Keeping this in one place is what lets Stage 5 move to rented NVIDIA GPUs without edits.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return cuda > mps > cpu, unless `prefer` (or $PT_DEVICE) names one explicitly."""
    choice = prefer or os.environ.get("PT_DEVICE")
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 0) -> None:
    """Fix every random source we use, so a rerun gives the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:
        pass
