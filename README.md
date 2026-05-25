# robustrolling

[![R package check](https://github.com/IgorPtak/rolling_window/actions/workflows/r_check.yml/badge.svg)](https://github.com/IgorPtak/rolling_window/actions/workflows/r_check.yml)
[![C++ tests](https://github.com/IgorPtak/rolling_window/actions/workflows/cpp_test.yml/badge.svg)](https://github.com/IgorPtak/rolling_window/actions/workflows/cpp_test.yml)
[![Python package](https://github.com/IgorPtak/rolling_window/actions/workflows/python.yml/badge.svg)](https://github.com/IgorPtak/rolling_window/actions/workflows/python.yml)

High-performance rolling window metrics for R and Python, implemented in C++17.

---

## Overview

`robustrolling` provides numerically stable, memory-efficient rolling window
algorithms built in C++17 and exposed to both R and Python. All algorithms:

- run in **O(1) time per element** (O(log n) for median),
- handle **NaN / NA transparently**,
- support a **`min_periods`** parameter (pandas-compatible semantics),
- share a common **CRTP base** (`RollingMetric<Derived>`) — zero virtual
  dispatch, flat ring-buffer memory layout,
- are compiled with **`-O3 -flto`** and include **ARM NEON / AVX2 SIMD** paths
  for `rolling_mean`.

---

## Features

| C++ class              | Algorithm                                    | Time           | R API                                                       | Python class           |
| ---------------------- | -------------------------------------------- | -------------- | ----------------------------------------------------------- | ---------------------- |
| `SlidingMean`          | Prefix sum + SIMD (ARM NEON / AVX2)          | O(n) batch     | `rolling_mean()`                                            | `SlidingMean`          |
| `SlidingWelfordRing`   | Welford online + ring buffer                 | O(1)           | `rolling_variance()` (`method="stable"`)                    | `SlidingWelford`       |
| `SlidingMomentsPrefix` | Prefix sums of raw moments                   | O(n) batch     | `rolling_variance/skewness/kurtosis()` (`method="fast"`)    | `SlidingMomentsPrefix` |
| `MonotonicMax`         | Monotonic deque                              | O(1) amortised | `rolling_max()`                                             | `MonotonicMax`         |
| `MonotonicMin`         | Monotonic deque                              | O(1) amortised | `rolling_min()`                                             | `MonotonicMin`         |
| `MultisetMedian`       | `std::multiset` dual-iterator                | O(log n)       | `rolling_median()`                                          | `MultisetMedian`       |
| `SlidingMoments`       | Terriberry's 4th-moment recurrence           | O(1)           | `rolling_skewness/kurtosis()` (`method="stable"`)           | `SlidingMoments`       |
| `SlidingCovariance`    | Welford 2D online                            | O(1)           | `rolling_cov()` `rolling_cor()`                             | `SlidingCovariance`    |

---

## Installation

### R

```r
remotes::install_github("IgorPtak/rolling_window")
```

Or build from source:

```bash
git clone https://github.com/IgorPtak/rolling_window.git
cd rolling_window
make r-build
```

Requires: R ≥ 4.0, a C++17 compiler.

### Python

```bash
git clone https://github.com/IgorPtak/rolling_window.git
cd rolling_window
pip install py_package/
```

With pandas support:

```bash
pip install "py_package/[pandas]"
```

Requires: Python ≥ 3.10, a C++17 compiler, pybind11.

---

## Usage

### R

```r
library(robustrolling)

x <- as.double(c(1, 3, 2, 5, 4))
```

**Max / Min**

```r
rolling_max(x, 3L)
#> [1] NA NA  3  5  5

rolling_min(x, 3L)
#> [1] NA NA  1  2  2
```

**Median**

```r
rolling_median(x, 3L)
#> [1] NA NA  2  3  4
```

**Variance and mean**

```r
y <- as.double(c(1, 2, 3, 4, 5))

rolling_variance(y, 3L)
#> [1] NA NA  1  1  1

rolling_mean(y, 3L)
#> [1] NA NA  2  3  4
```

**Higher moments**

```r
rolling_skewness(y, 3L)
#> [1] NA NA  0  0  0

rolling_kurtosis(y, 4L)
#> [1] NA NA NA -1.2 -1.2
```

**Covariance and Pearson correlation**

```r
a <- as.double(c(1, 2, 3, 4, 5))
b <- as.double(c(2, 4, 6, 8, 10))

rolling_cov(a, b, 3L)
#> [1] NA NA  2  2  2

rolling_cor(a, b, 3L)
#> [1] NA NA  1  1  1
```

**`min_periods` — require fewer observations**

```r
rolling_max(x, 3L, min_periods = 1L)
#> [1]  1  3  3  5  5
```

**Fast methods — prefix-sum acceleration**

`method = "fast"` uses prefix sums of raw moments instead of the online
Welford/Terriberry algorithm. It is 2–4x faster on large arrays but
susceptible to catastrophic cancellation for very large values with small
variance. Use it when numerical precision is not critical.

```r
rolling_variance(y, 3L, method = "fast")
rolling_skewness(y, 3L, method = "fast")
rolling_kurtosis(y, 4L, method = "fast")
```

`assume_finite = TRUE` enables the SIMD fast path for `rolling_mean` when the
input is guaranteed to contain no `NA` values:

```r
rolling_mean(y, 3L, assume_finite = TRUE)
```

---

### Python — high-level API

All functions accept `np.ndarray` and `pd.Series`:

```python
import numpy as np
import robustrolling as rr

x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

rr.rolling_max(x, 3)
# array([nan, nan,  3.,  5.,  5.])

rr.rolling_min(x, 3)
# array([nan, nan,  1.,  2.,  2.])

rr.rolling_median(x, 3)
# array([nan, nan,  2.,  3.,  4.])
```

```python
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

rr.rolling_variance(y, 3)
# array([nan, nan,  1.,  1.,  1.])

rr.rolling_mean(y, 3)
# array([nan, nan,  2.,  3.,  4.])

rr.rolling_skewness(y, 3)
# array([nan, nan,  0.,  0.,  0.])

rr.rolling_kurtosis(y, 4)
# array([nan, nan,  nan, -1.2, -1.2])
```

```python
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

rr.rolling_cov(a, b, 3)
# array([nan, nan,  2.,  2.,  2.])

rr.rolling_cor(a, b, 3)
# array([nan, nan,  1.,  1.,  1.])
```

**Fast methods**

```python
# 2–4x faster, less numerically stable
rr.rolling_variance(y, 3, method="fast")
rr.rolling_skewness(y, 3, method="fast")
rr.rolling_kurtosis(y, 4, method="fast")

# SIMD mean — assumes no NaN in input
rr.rolling_mean(y, 3, assume_finite=True)
```

**pandas Series — index and name are preserved:**

```python
import pandas as pd

prices = pd.Series(
    [100.0, 102.0, 98.0, 105.0, 103.0],
    index=pd.date_range("2024-01-01", periods=5),
    name="close",
)

rr.rolling_max(prices, 3)
# 2024-01-01      NaN
# 2024-01-02      NaN
# 2024-01-03    102.0
# 2024-01-04    105.0
# 2024-01-05    105.0
# Freq: D, Name: close, dtype: float64
```

### Python — low-level C++ classes

Direct access to the engine objects for incremental (streaming) use:

```python
from robustrolling import MonotonicMax, SlidingMoments, SlidingCovariance
import numpy as np

# Streaming — one value at a time
engine = MonotonicMax(3)
for v in [1.0, 3.0, 2.0, 5.0]:
    engine.update(v)
    print(engine.get_value())
# 1.0 → 3.0 → 3.0 → 5.0

# Batch — zero-copy NumPy buffer
engine2 = SlidingMoments(3)
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(engine2.process_skewness_batch(x))
# [nan, nan, 0., 0., 0.]

# Covariance engine
cov_engine = SlidingCovariance(3)
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([2.0, 4.0, 6.0, 8.0])
print(cov_engine.process_covariance_batch(a, b))
# [nan, nan, 2., 2.]
```

---

## Performance

Benchmarked on Apple M-series (ARM), window = 100, n = 1 000 000.

### Python vs pandas

| Function             | robustrolling | pandas   | speedup  |
| -------------------- | ------------- | -------- | -------- |
| `rolling_mean`       | 3.1 ms        | 4.4 ms   | **1.4x** |
| `rolling_max`        | 11.1 ms       | 11.7 ms  | 1.1x     |
| `rolling_min`        | 11.2 ms       | 12.2 ms  | 1.1x     |
| `rolling_median`     | 106 ms        | 233 ms   | **2.2x** |
| `rolling_variance`   | 15.2 ms       | 9.6 ms   | 0.6x     |
| `rolling_skewness`   | 14.0 ms       | 9.1 ms   | 0.6x     |
| `rolling_kurtosis`   | 14.3 ms       | 9.2 ms   | 0.6x     |
| `rolling_cov`        | 14.8 ms       | 18.2 ms  | **1.2x** |
| `rolling_cor`        | 14.6 ms       | 36.7 ms  | **2.5x** |

### Python vs Polars

| Function             | robustrolling | Polars   | speedup  |
| -------------------- | ------------- | -------- | -------- |
| `rolling_mean`       | 3.1 ms        | 8.0 ms   | **2.6x** |
| `rolling_max`        | 11.1 ms       | 11.4 ms  | 1.0x     |
| `rolling_min`        | 11.0 ms       | 11.6 ms  | 1.1x     |
| `rolling_median`     | 106 ms        | 40.8 ms  | 0.4x     |
| `rolling_variance`   | 15.7 ms       | 16.2 ms  | 1.0x     |
| `rolling_skewness`   | 13.9 ms       | 16.0 ms  | **1.2x** |
| `rolling_kurtosis`   | 14.3 ms       | 15.6 ms  | 1.1x     |

### Python — stable vs fast

| Function               | stable   | fast     | speedup  |
| ---------------------- | -------- | -------- | -------- |
| `mean` (assume_finite) | 3.2 ms   | 0.73 ms  | **4.4x** |
| `variance`             | 15.2 ms  | 3.9 ms   | **3.9x** |
| `skewness`             | 13.9 ms  | 10.0 ms  | **1.4x** |
| `kurtosis`             | 14.4 ms  | 7.6 ms   | **1.9x** |

### R vs slider vs RcppRoll

| Function             | robustrolling | slider     | RcppRoll  | vs slider  | vs RcppRoll |
| -------------------- | ------------- | ---------- | --------- | ---------- | ----------- |
| `rolling_max`        | 15.1 ms       | 338 ms     | 175 ms    | **22x**    | **12x**     |
| `rolling_min`        | 14.9 ms       | 350 ms     | 175 ms    | **24x**    | **12x**     |
| `rolling_mean`       | 3.1 ms        | 1 523 ms   | 37.4 ms   | **487x**   | **12x**     |
| `rolling_variance`   | 16.0 ms       | 2 477 ms   | 304 ms    | **154x**   | **19x**     |
| `rolling_median`     | 112 ms        | 10 084 ms  | 1 938 ms  | **90x**    | **17x**     |

### R — stable vs fast

| Function               | stable   | fast     | speedup  |
| ---------------------- | -------- | -------- | -------- |
| `mean` (assume_finite) | 3.2 ms   | 0.78 ms  | **4.0x** |
| `variance`             | 16.2 ms  | 4.1 ms   | **4.0x** |
| `skewness`             | 14.5 ms  | 10.3 ms  | **1.4x** |
| `kurtosis`             | 14.4 ms  | 7.8 ms   | **1.8x** |

---

## Architecture

The C++ core uses **CRTP (Curiously Recurring Template Pattern)** to share a
common interface across all algorithm classes without virtual dispatch:

```
RollingMetric<Derived>
├── SlidingMean          — prefix sum + ARM NEON / AVX2 SIMD
├── MonotonicMax         — monotonic deque (max)
├── MonotonicMin         — monotonic deque (min)
├── MultisetMedian       — std::multiset + dual-iterator median tracking
├── SlidingWelfordRing   — Welford variance + ring buffer eviction
├── SlidingMoments       — Terriberry's 4th-moment recurrence
└── SlidingCovariance    — 2D Welford for covariance and Pearson correlation

SlidingMomentsPrefix     — stateless batch engine (prefix sums of raw moments)
```

**Bindings:**

| Language | Technology                       | Notes                      |
| -------- | -------------------------------- | -------------------------- |
| R        | Pure R/C API (`.Call`)           | No Rcpp dependency         |
| Python   | pybind11 + NumPy buffer protocol | Zero-copy batch processing |

---

## Development

### Requirements

| Tool         | Version                                   |
| ------------ | ----------------------------------------- |
| C++ compiler | C++17 (GCC ≥ 9, Clang ≥ 10, MSVC ≥ 2019) |
| CMake        | ≥ 3.14                                    |
| R            | ≥ 4.0                                     |
| Python       | ≥ 3.10                                    |

### Build and test

```bash
# C++ unit tests (gtest)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_core --parallel
ctest --test-dir build --output-on-failure

# R package
make r-build   # sync headers + roxygen + R CMD INSTALL
make r-test    # tinytest

# Python package
make py-build  # editable install
make py-test   # pytest

# Benchmarks
Rscript benchmarks/bench_r.R
python benchmarks/bench_python.py
```

---

## License

MIT © Igor Ptak
