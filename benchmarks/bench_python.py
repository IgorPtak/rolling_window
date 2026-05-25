"""
Benchmark: robustrolling vs pandas rolling functions + stable vs fast.

Usage:
    pip install pandas
    python benchmarks/bench_python.py
"""

import time
import numpy as np
import pandas as pd
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
    y = RNG.standard_normal(n)
    s = pd.Series(x)
    t = pd.Series(y)
    return x, y, s, t


def run_vs_pandas(n: int) -> list[dict]:
    x, y, s, t = make_data(n)
    w = WINDOW
    roll = s.rolling(w)

    cases = [
        ("rolling_max",      lambda: rr.rolling_max(x, w),      lambda: roll.max()),
        ("rolling_min",      lambda: rr.rolling_min(x, w),      lambda: roll.min()),
        ("rolling_mean",     lambda: rr.rolling_mean(x, w),     lambda: roll.mean()),
        ("rolling_variance", lambda: rr.rolling_variance(x, w), lambda: roll.var()),
        ("rolling_median",   lambda: rr.rolling_median(x, w),   lambda: roll.median()),
        ("rolling_skewness", lambda: rr.rolling_skewness(x, w), lambda: roll.skew()),
        ("rolling_kurtosis", lambda: rr.rolling_kurtosis(x, w), lambda: roll.kurt()),
        ("rolling_cov",      lambda: rr.rolling_cov(x, y, w),   lambda: roll.cov(t)),
        ("rolling_cor",      lambda: rr.rolling_cor(x, y, w),   lambda: roll.corr(t)),
    ]

    results = []
    for name, our_fn, pd_fn in cases:
        our_ms = bench(our_fn)
        pd_ms = bench(pd_fn)
        results.append({"name": name, "our_ms": our_ms, "pd_ms": pd_ms,
                        "speedup": pd_ms / our_ms})
    return results


def run_stable_vs_fast(n: int) -> list[dict]:
    x, _y, _s, _t = make_data(n)
    w = WINDOW

    cases = [
        ("mean (assume_finite)",
         lambda: rr.rolling_mean(x, w),
         lambda: rr.rolling_mean(x, w, assume_finite=True)),
        ("variance",
         lambda: rr.rolling_variance(x, w),
         lambda: rr.rolling_variance(x, w, method="fast")),
        ("skewness",
         lambda: rr.rolling_skewness(x, w),
         lambda: rr.rolling_skewness(x, w, method="fast")),
        ("kurtosis",
         lambda: rr.rolling_kurtosis(x, w),
         lambda: rr.rolling_kurtosis(x, w, method="fast")),
    ]

    results = []
    for name, stable_fn, fast_fn in cases:
        stable_ms = bench(stable_fn)
        fast_ms = bench(fast_fn)
        results.append({"name": name, "stable_ms": stable_ms, "fast_ms": fast_ms,
                        "speedup": stable_ms / fast_ms})
    return results


def flag(v: float) -> str:
    return "x" if v >= 1.0 else " "


def print_vs_pandas(n: int, rows: list[dict]) -> None:
    print(f"\n  n = {n:,}   window = {WINDOW}   (median of {REPS} runs)")
    print(f"  {'Function':<22} {'robustrolling':>14} {'pandas':>10} {'speedup':>9}")
    print("  " + "-" * 59)
    for r in rows:
        print(
            f"  {r['name']:<22} {r['our_ms']:>11.2f} ms"
            f"  {r['pd_ms']:>7.2f} ms"
            f"  {r['speedup']:>6.2f}x {flag(r['speedup'])}"
        )


def print_stable_vs_fast(n: int, rows: list[dict]) -> None:
    print(f"\n  n = {n:,}   window = {WINDOW}   (median of {REPS} runs)")
    print(f"  {'Function':<22} {'stable':>12} {'fast':>10} {'speedup':>9}")
    print("  " + "-" * 57)
    for r in rows:
        print(
            f"  {r['name']:<22} {r['stable_ms']:>9.2f} ms"
            f"  {r['fast_ms']:>7.2f} ms"
            f"  {r['speedup']:>6.2f}x {flag(r['speedup'])}"
        )


if __name__ == "__main__":
    print("robustrolling vs pandas — rolling window benchmark")
    print("=" * 59)
    for n in SIZES:
        rows = run_vs_pandas(n)
        print_vs_pandas(n, rows)

    print("\n\nstable vs fast — prefix-sum acceleration")
    print("=" * 59)
    for n in SIZES:
        rows = run_stable_vs_fast(n)
        print_stable_vs_fast(n, rows)

    print()
