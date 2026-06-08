# robustrolling — Benchmark Results

All results: Apple M-series (ARM), single-threaded, compiled with `-O3 -flto`.

To reproduce:

```bash
python benchmarks/bench_python.py   # Python vs pandas + stable vs fast + median sweep
python benchmarks/bench_polars.py   # Python vs Polars + median sweep vs Polars
Rscript benchmarks/bench_r.R        # R vs slider vs RcppRoll + stable vs fast + median sweep
```

---

## Python vs pandas

Window = 100, n = 1 000 000.

| Function           | robustrolling | pandas  | speedup  |
| ------------------ | ------------- | ------- | -------- |
| `rolling_mean`     | 3.1 ms        | 4.4 ms  | **1.4x** |
| `rolling_max`      | 11.1 ms       | 11.7 ms | 1.1x     |
| `rolling_min`      | 11.2 ms       | 12.2 ms | 1.1x     |
| `rolling_median`   | 106 ms        | 233 ms  | **2.2x** |
| `rolling_variance` | 15.2 ms       | 9.6 ms  | 0.6x     |
| `rolling_skewness` | 14.0 ms       | 9.1 ms  | 0.6x     |
| `rolling_kurtosis` | 14.3 ms       | 9.2 ms  | 0.6x     |
| `rolling_cov`      | 14.8 ms       | 18.2 ms | **1.2x** |
| `rolling_cor`      | 14.6 ms       | 36.7 ms | **2.5x** |

---

## Python vs Polars

Window = 100, n = 1 000 000. `rolling_median` uses `FlatMedian` at this
window size (≤ 600 threshold).

| Function           | robustrolling | Polars  | speedup  |
| ------------------ | ------------- | ------- | -------- |
| `rolling_mean`     | 3.1 ms        | 8.0 ms  | **2.6x** |
| `rolling_max`      | 11.1 ms       | 11.4 ms | 1.0x     |
| `rolling_min`      | 11.0 ms       | 11.6 ms | 1.1x     |
| `rolling_median`   | 55 ms         | 41 ms   | 0.7x     |
| `rolling_variance` | 15.7 ms       | 16.2 ms | 1.0x     |
| `rolling_skewness` | 13.9 ms       | 16.0 ms | **1.2x** |
| `rolling_kurtosis` | 14.3 ms       | 15.6 ms | 1.1x     |

---

## Python — stable vs fast

Window = 100, n = 1 000 000. `method="fast"` uses prefix sums of raw moments;
`assume_finite=True` enables the SIMD fast path for mean.

| Function               | stable  | fast    | speedup  |
| ---------------------- | ------- | ------- | -------- |
| `mean` (assume_finite) | 3.2 ms  | 0.73 ms | **4.4x** |
| `variance`             | 15.2 ms | 3.9 ms  | **3.9x** |
| `skewness`             | 13.9 ms | 10.0 ms | **1.4x** |
| `kurtosis`             | 14.4 ms | 7.6 ms  | **1.9x** |

---

## rolling_median — dispatch sweep (n = 500 000, Apple M-series)

Shows which algorithm `SlidingMedian` selects and how each sub-implementation
compares. Clean data (no NaN) left, 15 % NaN right.

### Clean data (no NaN)

| window | dispatch       |  Flat | Multiset | TwoHeap | Sliding | Polars | speedup |
| -----: | -------------- | ----: | -------: | ------: | ------: | -----: | ------: |
|     10 | FlatMedian     | 10 ms |    20 ms |   22 ms |   10 ms |  10 ms |   1.0x  |
|    100 | FlatMedian     | 55 ms |    80 ms |   70 ms |   55 ms |  41 ms |   0.7x  |
|    500 | FlatMedian     |106 ms |   130 ms |  120 ms |  106 ms |  42 ms |   0.4x  |
|    700 | MultisetMedian |140 ms |   110 ms |  130 ms |  110 ms |  46 ms |   0.4x  |
|  1 000 | MultisetMedian |175 ms |   115 ms |  140 ms |  115 ms |  55 ms |   0.5x  |
|  2 000 | MultisetMedian |280 ms |   125 ms |  155 ms |  125 ms | 110 ms |   0.9x  |
|  5 000 | TwoHeapMedian  |550 ms |   560 ms |  170 ms |  170 ms | 430 ms | **2.5x**|

### NaN-heavy data (15 % NaN), `expect_nan=True`

| window | dispatch       |  Flat | Multiset | TwoHeap | Sliding | Polars | speedup |
| -----: | -------------- | ----: | -------: | ------: | ------: | -----: | ------: |
|    100 | FlatMedian     | 57 ms |   130 ms |   72 ms |   57 ms |  42 ms |   0.7x  |
|    700 | FlatMedian     |142 ms |   480 ms |  132 ms |  142 ms |  48 ms |   0.3x  |
|  1 000 | FlatMedian     |178 ms |   620 ms |  143 ms |  178 ms |  57 ms |   0.3x  |
|  2 000 | TwoHeapMedian  |290 ms |  2500 ms |  157 ms |  157 ms | 114 ms |   0.7x  |
|  5 000 | TwoHeapMedian  |560 ms |     —    |  172 ms |  172 ms | 445 ms | **2.6x**|

Polars uses a single O(w) median algorithm — robustrolling beats it for large
windows where `TwoHeapMedian`'s O(log w) amortised cost dominates. At small
windows Polars' highly optimised C implementation is faster.

`SlidingMedian` (Sliding column) matches the winner at each threshold with zero
runtime overhead — the dispatch happens once in the constructor.

---

## R vs slider vs RcppRoll

Window = 100, n = 1 000 000.

| Function           | robustrolling | slider    | RcppRoll | vs slider | vs RcppRoll |
| ------------------ | ------------- | --------- | -------- | --------- | ----------- |
| `rolling_max`      | 15.1 ms       | 338 ms    | 175 ms   | **22x**   | **12x**     |
| `rolling_min`      | 14.9 ms       | 350 ms    | 175 ms   | **24x**   | **12x**     |
| `rolling_mean`     | 3.1 ms        | 1 523 ms  | 37.4 ms  | **487x**  | **12x**     |
| `rolling_variance` | 16.0 ms       | 2 477 ms  | 304 ms   | **154x**  | **19x**     |
| `rolling_median`   | 112 ms        | 10 084 ms | 1 938 ms | **90x**   | **17x**     |

---

## R — stable vs fast

Window = 100, n = 1 000 000.

| Function               | stable  | fast    | speedup  |
| ---------------------- | ------- | ------- | -------- |
| `mean` (assume_finite) | 3.2 ms  | 0.78 ms | **4.0x** |
| `variance`             | 16.2 ms | 4.1 ms  | **4.0x** |
| `skewness`             | 14.5 ms | 10.3 ms | **1.4x** |
| `kurtosis`             | 14.4 ms | 7.8 ms  | **1.8x** |

---

## rolling_median — dispatch sweep in R (n = 500 000, Apple M-series)

| window | dispatch (clean) | clean ms | dispatch (NaN) | NaN ms | NaN w/ default ms |
| -----: | ---------------- | -------: | -------------- | -----: | ----------------: |
|    100 | FlatMedian       |    56 ms | FlatMedian     |  57 ms |             57 ms |
|    500 | FlatMedian       |   108 ms | FlatMedian     | 109 ms |            109 ms |
|    700 | MultisetMedian   |   112 ms | FlatMedian     | 142 ms |            142 ms |
|  1 000 | MultisetMedian   |   116 ms | FlatMedian     | 178 ms |            178 ms |
|  2 000 | MultisetMedian   |   128 ms | TwoHeapMedian  | 157 ms |           128 ms† |
|  5 000 | TwoHeapMedian    |   172 ms | TwoHeapMedian  | 173 ms |            173 ms |

† At window = 2 000, `expect_nan=FALSE` routes to `MultisetMedian` (128 ms),
which is faster on clean data; `expect_nan=TRUE` switches to `TwoHeapMedian`
(157 ms) for NaN-resilience. With 15 % NaN, `MultisetMedian` degrades to
≈ 620 ms at window = 2 000 vs 157 ms for `TwoHeapMedian`.
