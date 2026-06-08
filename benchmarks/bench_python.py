"""
Benchmark: robustrolling vs pandas rolling functions + stable vs fast
         + rolling_median dispatch sweep across window sizes.

Usage:
    pip install pandas
    python benchmarks/bench_python.py
"""

import time
import numpy as np
import pandas as pd
import robustrolling as rr
import robust_rolling_core as rrc

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
        ("rolling_max",      lambda: rr.rolling_max(x, w),                    lambda: roll.max()),
        ("rolling_min",      lambda: rr.rolling_min(x, w),                    lambda: roll.min()),
        ("rolling_mean",     lambda: rr.rolling_mean(x, w),                   lambda: roll.mean()),
        ("rolling_variance", lambda: rr.rolling_variance(x, w),               lambda: roll.var()),
        ("rolling_median",   lambda: rr.rolling_median(x, w),                 lambda: roll.median()),
        ("rolling_skewness", lambda: rr.rolling_skewness(x, w),               lambda: roll.skew()),
        ("rolling_kurtosis", lambda: rr.rolling_kurtosis(x, w),               lambda: roll.kurt()),
        ("rolling_cov",      lambda: rr.rolling_cov(x, y, w),                 lambda: roll.cov(t)),
        ("rolling_cor",      lambda: rr.rolling_cor(x, y, w),                 lambda: roll.corr(t)),
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


MEDIAN_WINDOWS = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, 5000]
MEDIAN_N = 500_000
MEDIAN_NAN_FRAC = 0.15


def run_median_sweep() -> list[dict]:
    """Per-window-size comparison of FlatMedian / MultisetMedian / TwoHeapMedian
    vs the SlidingMedian dispatcher, on clean and NaN-heavy data."""
    rng = np.random.default_rng(42)
    data_clean = rng.standard_normal(MEDIAN_N)
    data_nan = data_clean.copy()
    data_nan[rng.random(MEDIAN_N) < MEDIAN_NAN_FRAC] = np.nan

    # Dispatch thresholds (must match SlidingMedian.hpp constants)
    FLAT_CLEAN = 600
    HEAP_CLEAN = 2000
    FLAT_NAN   = 1500

    def which_algo(w: int, nan: bool) -> str:
        if nan:
            return "FlatMedian" if w <= FLAT_NAN else "TwoHeapMedian"
        if w <= FLAT_CLEAN:
            return "FlatMedian"
        if w <= HEAP_CLEAN:
            return "MultisetMedian"
        return "TwoHeapMedian"

    results = []
    for w in MEDIAN_WINDOWS:
        row: dict = {"window": w}
        for label, data, nan_hint in [("clean", data_clean, False),
                                      ("nan15", data_nan,   True)]:
            engines = {
                "Flat":     lambda _w=w: rrc.FlatMedian(_w),
                "Multiset": lambda _w=w: rrc.MultisetMedian(_w),
                "TwoHeap":  lambda _w=w: rrc.TwoHeapMedian(_w),
                "Sliding":  lambda _w=w, _h=nan_hint: rrc.SlidingMedian(_w, _h),
            }
            for name, factory in engines.items():
                row[f"{label}_{name}_ms"] = bench(
                    lambda _d=data, _f=factory: _f().process_batch(_d),
                    reps=3,
                )
            row[f"{label}_algo"] = which_algo(w, nan_hint)
        results.append(row)
    return results


def print_median_sweep(rows: list[dict]) -> None:
    hdr = f"  {'window':>6}  {'dispatches to':>15}  {'Flat':>8}  {'Multiset':>9}  {'TwoHeap':>9}  {'Sliding':>9}"
    for label, title in [("clean", "Clean data (no NaN)"), ("nan15", "NaN-heavy data (15% NaN)")]:
        print(f"\n  {title}")
        print(hdr)
        print("  " + "-" * 73)
        for r in rows:
            print(
                f"  {r['window']:>6}  {r[f'{label}_algo']:>15}"
                f"  {r[f'{label}_Flat_ms']:>6.1f} ms"
                f"  {r[f'{label}_Multiset_ms']:>7.1f} ms"
                f"  {r[f'{label}_TwoHeap_ms']:>7.1f} ms"
                f"  {r[f'{label}_Sliding_ms']:>7.1f} ms"
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

    print("\n\nrolling_median dispatch sweep  (n = 500 000)")
    print("=" * 75)
    sweep = run_median_sweep()
    print_median_sweep(sweep)

    print()
