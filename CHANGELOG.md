# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-06-08

### Added

#### C++
- `TwoHeapMedian` — rolling median via max-heap / min-heap pair, O(log n);
  faster than `MultisetMedian` for large clean windows
- `FlatMedian` — rolling median via sorted array copy, O(k log k);
  optimal for small windows (k ≤ 600 clean, k ≤ 1500 with NaN)
- `SlidingMedian` — adaptive facade that selects `FlatMedian`,
  `MultisetMedian`, or `TwoHeapMedian` at construction time based on
  `window_size` and `expect_nan`; exposes unified CRTP interface

#### Python
- `rolling_median()` now backed by `SlidingMedian` (adaptive strategy)
  instead of `MultisetMedian`; gains `expect_nan` parameter

#### R
- `rolling_median()` now backed by `SlidingMedian`; gains `expect_nan`
  parameter

#### Benchmarks
- `bench_median_sweep.cxx` — C++ sweep across window sizes comparing all
  three median implementations
- Extended `bench_polars.py`, `bench_python.py`, `bench_r.R` with median
  sections
- `BENCHMARKS.md` — reference benchmark results

### Changed
- `rolling_median` (Python + R) delegates to `SlidingMedian` instead of
  `MultisetMedian` directly; existing call sites are unaffected

### Tests
- C++ test suites: `test_flat_median`, `test_sliding_median`,
  `test_two_heap_median` — cover correctness, memory layout, dtype
  handling, and fuzz scenarios
- Python tests rewritten for stricter dtype and robustness coverage
- R wrapper made optional in CMake; removed R dependency from C++ CI

---

## [0.2.0] - 2026-05-16

### Added

#### C++
- `SlidingMean` — new CRTP class with ring-buffer mean; `fast_mean_batch`
  handles NaN via two prefix arrays (`prefix_sum`, `prefix_count`) with an
  ARM NEON / AVX2 SIMD main loop; `fast_mean_batch_finite` uses a single
  prefix array with `inv_k` multiplication for NaN-free inputs
- `SlidingMomentsPrefix` — stateless batch engine computing variance,
  skewness, and kurtosis from prefix sums of raw moments (`Σx`, `Σx²`,
  `Σx³`, `Σx⁴`); 2–4× faster than Welford/Terriberry for large arrays
- `SlidingMoments::update_skewness_only()` — skips M4 computation for
  skewness-only batches
- `SlidingCovariance::current_size()` — exposes non-NaN pair count

#### Python
- `rolling_mean()` gains `assume_finite=False` — when `True`, uses the
  SIMD single-prefix path (no NaN overhead)
- `rolling_variance()`, `rolling_skewness()`, `rolling_kurtosis()` gain
  `method="stable"` (default, Welford/Terriberry) and `method="fast"`
  (prefix-sum, 2–4× faster, less numerically stable)
- `min_periods` masking moved from Python into C++ `process_batch_generic`;
  all metrics now apply the threshold inside the extension module
- `SlidingMomentsPrefix` exposed as a low-level Python class with
  `variance_batch`, `skewness_batch`, `kurtosis_batch`

#### R
- `rolling_mean()` gains `assume_finite = FALSE`
- `rolling_variance()`, `rolling_skewness()`, `rolling_kurtosis()` gain
  `method = "stable"` / `method = "fast"`
- `rolling_mean_c` migrated from `SlidingMoments` to `SlidingMean`
- New C entry points: `rolling_variance_fast_c`, `rolling_skewness_fast_c`,
  `rolling_kurtosis_fast_c`

#### Build
- Python extension compiled with `-O3 -flto` (Link-Time Optimisation)

### Changed
- `rolling_mean` (Python + R) now uses `SlidingMean` instead of
  `SlidingMoments`, reducing per-element work from 4 moments to 1

---

## [0.1.0] - 2026-05-09

### Added

#### C++
- `MonotonicMax` — rolling maximum via monotonic deque, O(1) amortised
- `MonotonicMin` — rolling minimum via monotonic deque, O(1) amortised
- `MultisetMedian` — rolling median via `std::multiset` with tracked iterator, O(log n)
- `SlidingWelford` — rolling variance via Welford online algorithm, O(1)
- `SlidingWelfordRing` — Welford variant using a ring buffer, O(1)
- `SlidingMoments` — Terriberry's 4th-moment recurrence for mean, skewness, kurtosis, O(1)
- `SlidingCovariance` — 2D Welford for rolling covariance and Pearson correlation, O(1)
- CRTP base class `RollingMetric<Derived>` providing shared `update()`,
  `process_batch()`, `get_value()`, `current_size()` interface

#### R package
- `rolling_max()`, `rolling_min()`, `rolling_median()`, `rolling_mean()`
- `rolling_variance()`, `rolling_skewness()`, `rolling_kurtosis()`
- `rolling_cov()`, `rolling_cor()`
- `min_periods` parameter on all functions (defaults to `window_size`)

#### Python package
- Low-level C++ bindings via pybind11 (`robust_rolling_core`)
- High-level Python API (`robustrolling`) matching the R surface
- Pandas integration: functions accept `pd.Series` and return `pd.Series`
  with index and name preserved

### Fixed
- Iterator bug in `MultisetMedian::update_impl()`: even-sized windows
  filled in descending order returned wrong median
- Segfault in `MultisetMedian` for `window_size = 2` during sliding phase
