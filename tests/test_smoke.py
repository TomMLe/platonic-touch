"""Smoke tests: fast, CPU-safe checks that the package imports and basic math works.

These run in CI on Linux too, so nothing here may require Apple hardware.
"""

import torch

from platonic_touch.device import get_device, seed_everything


def test_get_device_returns_a_usable_device():
    dev = get_device()
    x = torch.ones(3, device=dev)
    assert x.sum().item() == 3.0


def test_explicit_cpu_override():
    assert get_device("cpu").type == "cpu"


def test_seed_makes_runs_repeatable():
    seed_everything(42)
    a = torch.randn(5)
    seed_everything(42)
    b = torch.randn(5)
    assert torch.equal(a, b)
