"""Sanity-check the ML environment: MLX on the Apple GPU, and PyTorch on MPS.

Run with:  uv run python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
import time


def check_mlx() -> None:
    import mlx.core as mx

    print(f"MLX {mx.__version__}  default device: {mx.default_device()}")

    # 1) Tiny correctness check against a hand-computed answer.
    a = mx.array([[1.0, 2.0], [3.0, 4.0]])
    b = mx.array([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b  # MLX is lazy: this only builds a graph...
    mx.eval(c)  # ...and this actually runs it on the GPU.
    assert c.tolist() == [[19.0, 22.0], [43.0, 50.0]], c
    print("  matmul correctness: OK")

    # 2) Autograd: d/dx of sum(x^2) is 2x.
    grad = mx.grad(lambda x: (x**2).sum())(mx.array([1.0, 2.0, 3.0]))
    assert grad.tolist() == [2.0, 4.0, 6.0], grad
    print("  autograd: OK")

    # 3) A rough speed number, so you can compare against MPS below.
    x = mx.random.normal((2048, 2048))
    mx.eval(x @ x)  # warm-up (compiles the Metal kernel)
    t0 = time.perf_counter()
    for _ in range(10):
        mx.eval(x @ x)
    dt = (time.perf_counter() - t0) / 10
    print(f"  2048x2048 matmul: {dt * 1e3:.1f} ms  (~{2 * 2048**3 / dt / 1e12:.2f} TFLOP/s)")


def check_torch() -> None:
    import torch

    print(f"PyTorch {torch.__version__}  MPS available: {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        print("  MPS not available; PyTorch will run on CPU.")
        return
    dev = torch.device("mps")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=dev)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=dev)
    assert (a @ b).cpu().tolist() == [[19.0, 22.0], [43.0, 50.0]]
    print("  matmul correctness on MPS: OK")

    x = torch.randn(2048, 2048, device=dev)
    _ = x @ x
    torch.mps.synchronize()  # GPU work is async; wait before timing
    t0 = time.perf_counter()
    for _ in range(10):
        _ = x @ x
    torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / 10
    print(f"  2048x2048 matmul: {dt * 1e3:.1f} ms  (~{2 * 2048**3 / dt / 1e12:.2f} TFLOP/s)")


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} on {platform.machine()} ({platform.platform()})")
    if platform.machine() != "arm64":
        sys.exit("Not running natively on Apple Silicon (arm64). MLX needs a native arm64 Python.")
    check_mlx()
    check_torch()
    print("\nAll checks passed.")
