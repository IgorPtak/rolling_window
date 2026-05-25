"""
Benchmark: robustrolling vs Polars rolling functions + stable vs fast.

Usage:
    pip install polars
    python benchmarks/bench_polars.py
"""

import time
import numpy as np
import polars as pl
import robustrolling as rr

RNG = np.random.default_rng(42)

SIZES = [10_000, 100_000, 1_000_000]
WINDOW = 100
REPS = 10


def bench(fn, reps: int = REPS) -> float:
    """Return median wall time in milliseconds over `reps` runs."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1_000


def make_data(n: int):
    x = RNG.standard_normal(n)
    y = RNG.standard_normal(n)
    sx = pl.Series(x)
    sy = pl.Series(y)
    return x, y, sx, sy


def run_vs_polars(n: int) -> list[dict]:
    x, y, sx, sy = make_data(n)
    w = WINDOW

    cases = [
        ("rolling_max",      lambda: rr.rolling_max(x, w),      lambda: sx.rolling_max(w)),
        ("rolling_min",      lambda: rr.rolling_min(x, w),      lambda: sx.rolling_min(w)),
        ("rolling_mean",     lambda: rr.rolling_mean(x, w),     lambda: sx.rolling_mean(w)),
        ("rolling_variance", lambda: rr.rolling_variance(x, w), lambda: sx.rolling_var(w)),
        ("rolling_median",   lambda: rr.rolling_median(x, w),   lambda: sx.rolling_median(w)),
        ("rolling_skewness", lambda: rr.rolling_skewness(x, w), lambda: sx.rolling_skew(w)),
        ("rolling_kurtosis", lambda: rr.rolling_kurtosis(x, w), lambda: sx.rolling_kurtosis(w)),
    ]

    results = []
    for name, our_fn, pl_fn in cases:
        our_ms = bench(our_fn)
        pl_ms = bench(pl_fn)
        results.append({"name": name, "our_ms": our_ms, "pl_ms": pl_ms,
                        "speedup": pl_ms / our_ms})
    return results


def run_fast_vs_polars(n: int) -> list[dict]:
    x, _y, sx, _sy = make_data(n)
    w = WINDOW

    cases = [
        ("rolling_mean (SIMD)",       lambda: rr.rolling_mean(x, w, assume_finite=True), lambda: sx.rolling_mean(w)),
        ("rolling_variance (fast)",   lambda: rr.rolling_variance(x, w, method="fast"),  lambda: sx.rolling_var(w)),
        ("rolling_skewness (fast)",   lambda: rr.rolling_skewness(x, w, method="fast"),  lambda: sx.rolling_skew(w)),
        ("rolling_kurtosis (fast)",   lambda: rr.rolling_kurtosis(x, w, method="fast"),  lambda: sx.rolling_kurtosis(w)),
    ]

    results = []
    for name, our_fn, pl_fn in cases:
        our_ms = bench(our_fn)
        pl_ms = bench(pl_fn)
        results.append({"name": name, "our_ms": our_ms, "pl_ms": pl_ms,
                        "speedup": pl_ms / our_ms})
    return results


def flag(v: float) -> str:
    return "x" if v >= 1.0 else " "


def print_table(n: int, rows: list[dict], label: str) -> None:
    print(f"\n  n = {n:,}   window = {WINDOW}   (median of {REPS} runs)")
    print(f"  {'Function':<28} {'robustrolling':>14} {'polars':>10} {'speedup':>9}")
    print("  " + "-" * 65)
    for r in rows:
        print(
            f"  {r['name']:<28} {r['our_ms']:>11.2f} ms"
            f"  {r['pl_ms']:>7.2f} ms"
            f"  {r['speedup']:>6.2f}x {flag(r['speedup'])}"
        )


if __name__ == "__main__":
    print(f"robustrolling vs Polars {pl.__version__} — rolling window benchmark")
    print("=" * 65)

    print("\n--- stable (default) methods vs Polars ---")
    for n in SIZES:
        rows = run_vs_polars(n)
        print_table(n, rows, "stable")

    print("\n\n--- fast methods vs Polars ---")
    for n in SIZES:
        rows = run_fast_vs_polars(n)
        print_table(n, rows, "fast")

    print()
