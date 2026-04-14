"""
Benchmark: jnp.linalg.solve vs pinv+matmul for xreg ridge regression.

Run with:
    uv run python benchmarks/xreg_solve_vs_pinv.py

Not part of the test suite — for manual verification of the ~2x speedup claim.

Background
----------
xreg fits a ridge regression:  beta = (X^T X + λI)^{-1} X^T y

Old code used jnp.linalg.pinv (SVD-based, O(d^3) with a large constant).
New code uses jnp.linalg.solve (LU decomposition, O(d^3/3) — fewer flops, and
LAPACK's dgesv is more cache-friendly than the full SVD path used by pinv).
Both are numerically correct; solve is also more stable for well-conditioned systems.

Measured results (JAX 0.9.2, CPU, ridge=0.1):

  n=  50  d= 5  |  solve 0.007 ms  pinv 0.012 ms  →  1.75x faster
  n= 200  d=10  |  solve 0.007 ms  pinv 0.023 ms  →  3.12x faster
  n= 500  d=20  |  solve 0.010 ms  pinv 0.051 ms  →  5.25x faster
  n=1000  d=50  |  solve 0.035 ms  pinv 0.207 ms  →  5.86x faster

The "~2x" stated in the PR comment is the conservative lower bound (smallest
matrix, d=5). Speedup grows with d because SVD complexity scales worse than LU.
Max absolute error vs pinv is < 2e-7 across all sizes (well within float32 noise).

Why CPU?
--------
xreg covariate matrices are small (d = number of covariates, typically 5-50).
At these sizes GPU kernel-launch overhead dominates actual compute, making the
ratio noisy and unreliable. CPU isolates the algorithmic difference cleanly and
matches the GitHub Actions CI environment where this code actually runs.
"""

import time

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise SystemExit("JAX is not installed. Install with: uv sync --extra flax-cpu")

REPEATS = 200
WARMUP = 20
RIDGE = 0.1


def bench(fn, warmup: int, repeats: int) -> float:
    """Return median wall-clock time in milliseconds."""
    for _ in range(warmup):
        fn().block_until_ready()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn().block_until_ready()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def make_problem(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = jnp.array(rng.standard_normal((n, d)))
    y = jnp.array(rng.standard_normal(n))
    A = x.T @ x + RIDGE * jnp.eye(d)
    b = x.T @ y
    return A, b


def run(n: int, d: int):
    A, b = make_problem(n, d)
    # JIT-compile both variants so we measure steady-state, not trace time
    solve_fn = jax.jit(lambda: jnp.linalg.solve(A, b))
    pinv_fn = jax.jit(lambda: jnp.linalg.pinv(A, hermitian=True) @ b)

    t_solve = bench(solve_fn, WARMUP, REPEATS)
    t_pinv = bench(pinv_fn, WARMUP, REPEATS)
    ratio = t_pinv / t_solve

    # Verify correctness while we're here
    beta_solve = np.array(jnp.linalg.solve(A, b))
    beta_pinv = np.array(jnp.linalg.pinv(A, hermitian=True) @ b)
    max_err = np.max(np.abs(beta_solve - beta_pinv))

    print(
        f"n={n:4d} d={d:2d} | solve {t_solve:6.3f} ms  pinv {t_pinv:6.3f} ms"
        f"  ratio {ratio:.2f}x  max_err {max_err:.2e}"
    )


if __name__ == "__main__":
    print(f"JAX {jax.__version__}  ridge={RIDGE}  warmup={WARMUP}  repeats={REPEATS}")
    print(f"{'':30s} solve       pinv      speedup  correctness")
    print("-" * 72)
    for n, d in [(50, 5), (200, 10), (500, 20), (1000, 50)]:
        run(n, d)
