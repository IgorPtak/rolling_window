"""
Benchmark: robustrolling vs Polars rolling functions (stable methods only).

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
    fn()  # warmup: prime caches before timing
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1_000


def make_data(n: int):
    x = RNG.standard_normal(n)
    return x, pl.Series(x)


def run_vs_polars(n: int) -> list[dict]:
    x, sx = make_data(n)
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


def flag(v: float) -> str:
    return "x" if v >= 1.0 else " "


def print_table(n: int, rows: list[dict]) -> None:
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
    for n in SIZES:
        rows = run_vs_polars(n)
        print_table(n, rows)

    print()
