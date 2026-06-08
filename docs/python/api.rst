Python API Reference
====================

All functions accept ``np.ndarray`` and ``pd.Series`` and return the same
type. The ``min_periods`` parameter follows pandas semantics: positions with
fewer valid observations than ``min_periods`` are set to ``nan``.

High-level functions
--------------------

.. automodule:: robustrolling
   :members: rolling_max, rolling_min, rolling_median, rolling_variance,
             rolling_mean, rolling_skewness, rolling_kurtosis,
             rolling_cov, rolling_cor
   :no-undoc-members:

Low-level classes
-----------------

Eleven C++ classes are exposed directly for streaming (one observation at a
time) or for computing multiple statistics in a single pass without calling
several high-level functions.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - Algorithm
     - Key methods
   * - :py:class:`~robustrolling.SlidingMean`
     - Prefix sum + SIMD
     - ``update``, ``get_mean``, ``process_batch``
   * - :py:class:`~robustrolling.MonotonicMax`
     - Monotonic deque
     - ``update``, ``get_max``, ``process_batch``
   * - :py:class:`~robustrolling.MonotonicMin`
     - Monotonic deque
     - ``update``, ``get_min``, ``process_batch``
   * - :py:class:`~robustrolling.SlidingMedian`
     - Auto-dispatch (Flat / Multiset / TwoHeap)
     - ``update``, ``get_median``, ``process_batch``
   * - :py:class:`~robustrolling.FlatMedian`
     - Sorted vector — best w ≤ 600 or NaN-heavy
     - ``update``, ``get_median``, ``process_batch``
   * - :py:class:`~robustrolling.MultisetMedian`
     - ``std::multiset`` — best 601–2 000, clean data
     - ``update``, ``get_median``, ``process_batch``
   * - :py:class:`~robustrolling.TwoHeapMedian`
     - Two heaps + lazy deletion — best w > 2 000 or NaN-heavy
     - ``update``, ``get_median``, ``process_batch``
   * - :py:class:`~robustrolling.SlidingWelford`
     - Welford + ring buffer
     - ``update``, ``get_variance``, ``process_batch``
   * - :py:class:`~robustrolling.SlidingMoments`
     - Terriberry 4th-moment
     - ``update``, ``get_mean``, ``get_skewness``, ``get_kurtosis``, ``process_skewness_batch``, ``process_kurtosis_batch``
   * - :py:class:`~robustrolling.SlidingCovariance`
     - 2-D Welford
     - ``update``, ``get_covariance``, ``get_correlation``, ``process_covariance_batch``
   * - :py:class:`~robustrolling.SlidingMomentsPrefix`
     - Prefix sums of raw moments (stateless batch)
     - ``variance_batch``, ``skewness_batch``, ``kurtosis_batch``

.. toctree::
   :hidden:

   low_level
