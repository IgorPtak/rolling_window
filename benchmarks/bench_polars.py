"""
Benchmark: robustrolling vs Polars rolling_median.
Uses process_batch for all engines (no Python-loop overhead).

Usage:
    python benchmarks/bench_polars.py
"""

import time
import numpy as np
import polars as pl
import robustrolling as rr
import robust_rolling_core as rrc

RNG = np.random.default_rng(42)

N          = 500_000
REPS       = 5
NAN_FRAC   = 0.15
WINDOW     = 100   # fixed window for the vs-Polars all-metrics table

WINDOWS = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, 5000]


def bench(fn, reps=REPS) -> float:
    """Return median wall-time in milliseconds over *reps* runs (1 warmup)."""
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1_000


def ns_per_elem(ms: float) -> float:
    return ms / N * 1e6


def flag(v: float) -> str:
    return "✓" if v >= 1.0 else " "


# ── 1. all-metrics comparison at fixed window ──────────────────────────────────

def run_vs_polars(n: int) -> list[dict]:
    rng  = np.random.default_rng(0)
    x    = rng.standard_normal(n)
    sx   = pl.Series(x)
    w    = WINDOW

    cases = [
        ("rolling_max",      lambda: rr.rolling_max(x, w),
                             lambda: sx.rolling_max(window_size=w)),
        ("rolling_min",      lambda: rr.rolling_min(x, w),
                             lambda: sx.rolling_min(window_size=w)),
        ("rolling_mean",     lambda: rr.rolling_mean(x, w),
                             lambda: sx.rolling_mean(window_size=w)),
        ("rolling_variance", lambda: rr.rolling_variance(x, w),
                             lambda: sx.rolling_var(window_size=w)),
        ("rolling_median",   lambda: rr.rolling_median(x, w),
                             lambda: sx.rolling_median(window_size=w)),
        ("rolling_skewness", lambda: rr.rolling_skewness(x, w),
                             lambda: sx.rolling_skew(window_size=w)),
        ("rolling_kurtosis", lambda: rr.rolling_kurtosis(x, w),
                             lambda: sx.rolling_kurtosis(window_size=w)),
    ]

    rows = []
    for name, our_fn, pl_fn in cases:
        our_ms = bench(our_fn)
        pl_ms  = bench(pl_fn)
        rows.append({"name": name, "our_ms": our_ms, "pl_ms": pl_ms,
                     "speedup": pl_ms / our_ms})
    return rows


def print_vs_polars(n: int, rows: list[dict]) -> None:
    print(f"\n  n = {n:,}   window = {WINDOW}   (median of {REPS} runs)")
    print(f"  {'Function':<22} {'robustrolling':>14} {'polars':>10} {'speedup':>9}")
    print("  " + "-" * 59)
    for r in rows:
        print(
            f"  {r['name']:<22} {r['our_ms']:>11.2f} ms"
            f"  {r['pl_ms']:>7.2f} ms"
            f"  {r['speedup']:>6.2f}x {flag(r['speedup'])}"
        )


# ── 2. rolling_median sweep vs Polars ─────────────────────────────────────────

def which_algo(w: int, nan: bool) -> str:
    """Mirror the SlidingMedian dispatch thresholds."""
    if nan:
        return "Flat" if w <= 1500 else "TwoHeap"
    if w <= 600:  return "Flat"
    if w <= 2000: return "Multiset"
    return "TwoHeap"


def run_median_sweep() -> list[dict]:
    data_clean = RNG.standard_normal(N)
    data_nan   = data_clean.copy()
    data_nan[RNG.random(N) < NAN_FRAC] = np.nan

    pl_clean = pl.Series(data_clean)
    pl_nan   = pl.Series(data_nan).fill_nan(None)   # Polars uses null, not NaN

    rows = []
    for w in WINDOWS:
        row: dict = {"window": w}

        for label, data, nan_hint, pl_s in [
            ("clean", data_clean, False, pl_clean),
            ("nan",   data_nan,   True,  pl_nan),
        ]:
            # individual engines via process_batch (pure C++, no Python loop)
            for name, factory in [
                ("Flat",     lambda _w=w: rrc.FlatMedian(_w)),
                ("Multiset", lambda _w=w: rrc.MultisetMedian(_w)),
                ("TwoHeap",  lambda _w=w: rrc.TwoHeapMedian(_w)),
            ]:
                row[f"{label}_{name}_ms"] = bench(
                    lambda _f=factory, _d=data: _f().process_batch(_d)
                )

            # SlidingMedian dispatcher
            row[f"{label}_Sliding_ms"] = bench(
                lambda _w=w, _h=nan_hint, _d=data:
                    rrc.SlidingMedian(_w, _h).process_batch(_d)
            )

            # high-level Python API (rr.rolling_median)
            row[f"{label}_rr_ms"] = bench(
                lambda _d=data, _w=w: rr.rolling_median(_d, _w)
            )

            # Polars
            row[f"{label}_polars_ms"] = bench(
                lambda _s=pl_s, _w=w: _s.rolling_median(window_size=_w)
            )

            row[f"{label}_algo"] = which_algo(w, nan_hint)

        rows.append(row)
    return rows


def print_median_sweep(rows: list[dict]) -> None:
    for label, title in [
        ("clean", "Clean data  (no NaN)"),
        ("nan",   "NaN-heavy   (15% NaN)"),
    ]:
        print(f"\n  {title}  —  n = {N:,}  (median of {REPS} runs, ns per element)")
        print(
            f"  {'win':>5}  {'dispatches':>9}"
            f"  {'Flat':>9}  {'Multiset':>9}  {'TwoHeap':>9}"
            f"  {'Sliding':>9}  {'rr.API':>9}  {'Polars':>9}"
            f"  {'speedup':>9}"
        )
        print("  " + "-" * 97)
        for r in rows:
            w    = r["window"]
            sl   = r[f"{label}_Sliding_ms"]
            po   = r[f"{label}_polars_ms"]
            su   = po / sl if sl > 0 else float("nan")
            fl   = ns_per_elem(r[f"{label}_Flat_ms"])
            mu   = ns_per_elem(r[f"{label}_Multiset_ms"])
            he   = ns_per_elem(r[f"{label}_TwoHeap_ms"])
            sl_n = ns_per_elem(sl)
            rr_n = ns_per_elem(r[f"{label}_rr_ms"])
            po_n = ns_per_elem(po)
            algo = r[f"{label}_algo"]
            print(
                f"  {w:>5}  {algo:>9}"
                f"  {fl:>7.2f} ns  {mu:>7.2f} ns  {he:>7.2f} ns"
                f"  {sl_n:>7.2f} ns  {rr_n:>7.2f} ns  {po_n:>7.2f} ns"
                f"  {su:>7.2f}x {flag(su)}"
            )

    print()
    print("  Sliding  = SlidingMedian.process_batch  (auto-dispatch, no Python overhead)")
    print("  rr.API   = rr.rolling_median             (high-level Python wrapper)")
    print("  speedup  = Polars ns / Sliding ns        (>1.0x means we are faster)")


# ── 3. best engine per window (heatmap-style summary) ─────────────────────────

def print_winner_summary(rows: list[dict]) -> None:
    engines = ["Flat", "Multiset", "TwoHeap", "Sliding"]
    print("\n  Best C++ engine vs Polars per window  (lower ns/elem is better)")
    print(f"  {'win':>5}  {'clean: best':>13}  {'ns/el':>8}  {'speedup':>8}"
          f"     {'nan: best':>13}  {'ns/el':>8}  {'speedup':>8}")
    print("  " + "-" * 80)
    for r in rows:
        w = r["window"]
        for label in ["clean", "nan"]:
            best_name = min(engines, key=lambda e, l=label, _r=r: _r[f"{l}_{e}_ms"])
            best_ms   = r[f"{label}_{best_name}_ms"]
            po_ms     = r[f"{label}_polars_ms"]
            su        = po_ms / best_ms if best_ms > 0 else float("nan")
            if label == "clean":
                c_str = f"{best_name:>13}  {ns_per_elem(best_ms):>7.2f} ns  {su:>7.2f}x {flag(su)}"
            else:
                n_str = f"{best_name:>13}  {ns_per_elem(best_ms):>7.2f} ns  {su:>7.2f}x {flag(su)}"
        print(f"  {w:>5}  {c_str}     {n_str}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"robustrolling vs Polars {pl.__version__}  —  rolling window benchmark")
    print("=" * 65)

    print("\n[1] All metrics at fixed window")
    print("=" * 65)
    for n in [10_000, 100_000, 1_000_000]:
        rows = run_vs_polars(n)
        print_vs_polars(n, rows)

    print(f"\n\n[2] rolling_median sweep vs Polars  (n = {N:,})")
    print("=" * 99)
    sweep = run_median_sweep()
    print_median_sweep(sweep)

    print("\n\n[3] Best engine summary")
    print("=" * 82)
    print_winner_summary(sweep)

    print()
