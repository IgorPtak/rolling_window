robustrolling
=============

.. raw:: html

   <p class="hero">
   High-performance rolling-window statistics for R and Python — eleven
   algorithms implemented in C++17, exposed through idiomatic bindings in
   both languages, with O(1) or O(log w) updates per element.
   </p>

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api
   r_api/index

Algorithm overview
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 26 12 18 18

   * - C++ class
     - Algorithm
     - Complexity
     - R function(s)
     - Python class / function(s)
   * - ``SlidingMean``
     - Prefix sum + SIMD
     - O(n) batch
     - ``rolling_mean``
     - ``rolling_mean``, ``SlidingMean``
   * - ``SlidingWelfordRing``
     - Welford online variance (ring buffer)
     - O(1)
     - ``rolling_variance`` (``method="stable"``)
     - ``rolling_variance``, ``SlidingWelford``
   * - ``SlidingMomentsPrefix``
     - Prefix sums of raw moments
     - O(n) batch
     - ``rolling_variance/skewness/kurtosis`` (``method="fast"``)
     - ``SlidingMomentsPrefix``
   * - ``MonotonicMax``
     - Monotonic deque maximum
     - O(1) amortised
     - ``rolling_max``
     - ``rolling_max``, ``MonotonicMax``
   * - ``MonotonicMin``
     - Monotonic deque minimum
     - O(1) amortised
     - ``rolling_min``
     - ``rolling_min``, ``MonotonicMin``
   * - ``SlidingMedian``
     - Auto-dispatches to one of three below
     - —
     - ``rolling_median`` (``expect_nan=``)
     - ``rolling_median``, ``SlidingMedian``
   * - ``FlatMedian``
     - Sorted ``std::vector`` + binary search
     - O(w) insert
     - —
     - ``FlatMedian``
   * - ``MultisetMedian``
     - ``std::multiset`` + tracked iterator
     - O(log w)
     - —
     - ``MultisetMedian``
   * - ``TwoHeapMedian``
     - Two heaps + lazy deletion
     - O(log w)
     - —
     - ``TwoHeapMedian``
   * - ``SlidingMoments``
     - Terriberry 4th-moment online algorithm
     - O(1)
     - ``rolling_skewness``, ``rolling_kurtosis`` (``method="stable"``)
     - ``rolling_skewness``, ``rolling_kurtosis``, ``SlidingMoments``
   * - ``SlidingCovariance``
     - 2-D Welford online covariance
     - O(1)
     - ``rolling_cov``, ``rolling_cor``
     - ``rolling_cov``, ``rolling_cor``, ``SlidingCovariance``

Performance
-----------

Benchmarked on Apple M-series (ARM), window = 100, n = 1 000 000
(median of 10 runs).

**Python vs pandas**

.. list-table::
   :header-rows: 1
   :widths: 28 18 14 12

   * - Function
     - robustrolling
     - pandas
     - speedup
   * - ``rolling_mean``
     - 3.1 ms
     - 4.4 ms
     - **1.4x**
   * - ``rolling_max``
     - 11.1 ms
     - 11.7 ms
     - 1.1x
   * - ``rolling_min``
     - 11.2 ms
     - 12.2 ms
     - 1.1x
   * - ``rolling_median``
     - 106 ms
     - 233 ms
     - **2.2x**
   * - ``rolling_variance``
     - 15.2 ms
     - 9.6 ms
     - 0.6x
   * - ``rolling_skewness``
     - 14.0 ms
     - 9.1 ms
     - 0.6x
   * - ``rolling_kurtosis``
     - 14.3 ms
     - 9.2 ms
     - 0.6x
   * - ``rolling_cov``
     - 14.8 ms
     - 18.2 ms
     - **1.2x**
   * - ``rolling_cor``
     - 14.6 ms
     - 36.7 ms
     - **2.5x**

**Python vs Polars**

``rolling_median`` uses ``FlatMedian`` at window ≤ 600; ``TwoHeapMedian``
wins at large windows where Polars' fixed O(w) algorithm loses ground.

.. list-table::
   :header-rows: 1
   :widths: 28 18 14 12

   * - Function
     - robustrolling
     - Polars
     - speedup
   * - ``rolling_mean``
     - 3.1 ms
     - 8.0 ms
     - **2.6x**
   * - ``rolling_max``
     - 11.1 ms
     - 11.4 ms
     - 1.0x
   * - ``rolling_min``
     - 11.0 ms
     - 11.6 ms
     - 1.1x
   * - ``rolling_median``
     - 55 ms
     - 41 ms
     - 0.7x (w=100)
   * - ``rolling_variance``
     - 15.7 ms
     - 16.2 ms
     - 1.0x
   * - ``rolling_skewness``
     - 13.9 ms
     - 16.0 ms
     - **1.2x**
   * - ``rolling_kurtosis``
     - 14.3 ms
     - 15.6 ms
     - 1.1x

**Python — stable vs fast (prefix-sum acceleration)**

.. list-table::
   :header-rows: 1
   :widths: 28 14 12 12

   * - Function
     - stable
     - fast
     - speedup
   * - ``mean`` (assume_finite)
     - 3.2 ms
     - 0.73 ms
     - **4.4x**
   * - ``variance``
     - 15.2 ms
     - 3.9 ms
     - **3.9x**
   * - ``skewness``
     - 13.9 ms
     - 10.0 ms
     - **1.4x**
   * - ``kurtosis``
     - 14.4 ms
     - 7.6 ms
     - **1.9x**

**R vs slider vs RcppRoll**

.. list-table::
   :header-rows: 1
   :widths: 22 16 14 14 12 14

   * - Function
     - robustrolling
     - slider
     - RcppRoll
     - vs slider
     - vs RcppRoll
   * - ``rolling_max``
     - 15.1 ms
     - 338 ms
     - 175 ms
     - **22x**
     - **12x**
   * - ``rolling_min``
     - 14.9 ms
     - 350 ms
     - 175 ms
     - **24x**
     - **12x**
   * - ``rolling_mean``
     - 3.1 ms
     - 1 523 ms
     - 37.4 ms
     - **487x**
     - **12x**
   * - ``rolling_variance``
     - 16.0 ms
     - 2 477 ms
     - 304 ms
     - **154x**
     - **19x**
   * - ``rolling_median``
     - 112 ms
     - 10 084 ms
     - 1 938 ms
     - **90x**
     - **17x**

**R — stable vs fast**

.. list-table::
   :header-rows: 1
   :widths: 28 14 12 12

   * - Function
     - stable
     - fast
     - speedup
   * - ``mean`` (assume_finite)
     - 3.2 ms
     - 0.78 ms
     - **4.0x**
   * - ``variance``
     - 16.2 ms
     - 4.1 ms
     - **4.0x**
   * - ``skewness``
     - 14.5 ms
     - 10.3 ms
     - **1.4x**
   * - ``kurtosis``
     - 14.4 ms
     - 7.8 ms
     - **1.8x**

Install for R
-------------

.. code-block:: r

   # Requires R ≥ 4.0 and a C++17 compiler
   install.packages("remotes")
   remotes::install_github("IgorPtak/rolling_window")

Install for Python
------------------

.. code-block:: bash

   # Requires Python ≥ 3.10 and a C++17 compiler
   pip install git+https://github.com/IgorPtak/rolling_window.git#subdirectory=py_package

   # Or clone and install locally:
   git clone https://github.com/IgorPtak/rolling_window.git
   pip install rolling_window/py_package/
