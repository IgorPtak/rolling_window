Low-level Classes
=================

.. py:currentmodule:: robustrolling

C++17 classes exposed via pybind11. Use them for streaming (one value at a
time) or to read multiple statistics from a single pass.

----

.. py:class:: SlidingMean(window_size)

   Rolling mean — prefix sum with optional ARM NEON / AVX2 SIMD, O(n) batch.

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_mean() -> float
   .. py:method:: process_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray

   .. code-block:: python

      import robust_rolling_core as rrc
      import numpy as np

      sm = rrc.SlidingMean(3)
      x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
      sm.process_batch(x)  # [1.0, 1.5, 2.0, 3.0, 4.0]

----

.. py:class:: MonotonicMax(window_size)

   Rolling maximum — monotonic deque, O(1) amortised.

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_max() -> float
   .. py:method:: process_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray

   .. code-block:: python

      mm = rrc.MonotonicMax(3)
      mm.update(1.0); mm.update(3.0); mm.update(2.0)
      mm.get_max()  # 3.0

----

.. py:class:: MonotonicMin(window_size)

   Rolling minimum — monotonic deque, O(1) amortised.

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_min() -> float
   .. py:method:: process_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray

----

Median classes
--------------

Four classes implement rolling median. :py:class:`SlidingMedian` is the
recommended default — it selects the fastest sub-implementation in the
constructor, with zero runtime dispatch overhead (``std::visit`` on
``std::variant``).

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Class
     - Algorithm
     - Best for
   * - :py:class:`SlidingMedian`
     - Auto-dispatcher
     - All cases — picks one of the three below
   * - :py:class:`FlatMedian`
     - Sorted ``std::vector``
     - Small windows (w ≤ 600) or NaN-heavy data
   * - :py:class:`MultisetMedian`
     - ``std::multiset`` + tracked iterator
     - Medium windows (601–2 000), clean data
   * - :py:class:`TwoHeapMedian`
     - Two heaps + lazy deletion
     - Large windows (w > 2 000) or NaN-heavy data

.. py:class:: SlidingMedian(window_size, expect_nan=False)

   Rolling median auto-dispatcher. Selects the fastest of
   :py:class:`FlatMedian`, :py:class:`MultisetMedian`, or
   :py:class:`TwoHeapMedian` based on *window_size* and *expect_nan* at
   construction time.

   Dispatch thresholds (windows > 2 000 always use ``TwoHeapMedian``):

   .. list-table::
      :header-rows: 1
      :widths: 18 20 22 22

      * - ``expect_nan``
        - w ≤ 600
        - 601–1 500
        - 1 501–2 000
      * - ``False``
        - ``FlatMedian``
        - ``MultisetMedian``
        - ``MultisetMedian``
      * - ``True``
        - ``FlatMedian``
        - ``FlatMedian``
        - ``TwoHeapMedian``

   :param window_size: int
   :param expect_nan: bool — hint that input contains many NaN values
                      (default ``False``)

   .. py:method:: update(value: float)
   .. py:method:: get_median() -> float
   .. py:method:: process_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray

   .. code-block:: python

      import numpy as np
      import robust_rolling_core as rrc

      x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

      rrc.SlidingMedian(3).process_batch(x)
      # array([1., 2., 2., 3., 4.])

      # NaN-heavy data — use expect_nan=True for large windows
      rrc.SlidingMedian(700, expect_nan=True).process_batch(x)

----

.. py:class:: FlatMedian(window_size)

   Rolling median — sorted ``std::vector`` with binary-search insertion and
   eviction.  O(w) insert/evict but cache-friendly; fastest for small
   windows (w ≤ 600) and for NaN-heavy streams where iterator tracking would
   degrade.

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_median() -> float
   .. py:method:: process_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray

----

.. py:class:: MultisetMedian(window_size)

   Rolling median — ``std::multiset`` (red-black tree) with a tracked
   ``mid_`` iterator.  O(log w) insert/evict; fastest for medium windows
   (601–2 000) on clean data.  Degrades significantly on NaN-heavy streams
   because iterator repositioning must scan the tree.

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_median() -> float
   .. py:method:: process_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray

----

.. py:class:: TwoHeapMedian(window_size)

   Rolling median — two heaps (max-heap for lower half, min-heap for upper
   half) with lazy deletion via a ``pending`` map.  O(log w) amortised;
   memory layout is less cache-friendly than ``FlatMedian`` but unaffected
   by NaN density, making it the best choice for large windows or NaN-heavy
   data.

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_median() -> float
   .. py:method:: process_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray

----

.. py:class:: SlidingWelford(window_size)

   Rolling sample variance (ddof=1) — Welford algorithm with ring buffer,
   O(1).

   :param window_size: int

   .. py:method:: update(value: float)
   .. py:method:: get_variance() -> float
   .. py:method:: process_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray

   .. code-block:: python

      sw = rrc.SlidingWelford(3)
      for v in [1., 2., 3., 4.]:
          sw.update(v)
      sw.get_variance()  # 1.0

----

.. py:class:: SlidingMoments(window_size)

   Rolling mean, skewness, and excess kurtosis — Terriberry's 4th-moment
   algorithm, O(1).  Requires ≥ 3 observations for skewness, ≥ 4 for
   kurtosis.

   :param window_size: int

   .. py:method:: update(x: float)
   .. py:method:: reset()
   .. py:method:: current_size() -> int
   .. py:method:: get_mean() -> float
   .. py:method:: get_skewness() -> float
   .. py:method:: get_kurtosis() -> float
   .. py:method:: process_mean_batch(x: numpy.ndarray, min_periods: int) -> numpy.ndarray
   .. py:method:: process_skewness_batch(x: numpy.ndarray, min_periods: int) -> numpy.ndarray
   .. py:method:: process_kurtosis_batch(x: numpy.ndarray, min_periods: int) -> numpy.ndarray

   Note: ``min_periods`` is a required positional argument in the
   ``process_*_batch`` methods (no default).

   .. code-block:: python

      sm = rrc.SlidingMoments(4)
      for v in [1., 2., 3., 4.]:
          sm.update(v)
      sm.get_mean(), sm.get_skewness(), sm.get_kurtosis()
      # (2.5, 0.0, -1.2)

      # Batch usage
      x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
      rrc.SlidingMoments(3).process_skewness_batch(x, 0)
      # [nan, nan, 0., 0., 0.]

----

.. py:class:: SlidingCovariance(window_size)

   Rolling sample covariance and Pearson correlation — 2-D Welford
   algorithm, O(1).

   :param window_size: int

   .. py:method:: update(x: float, y: float)
   .. py:method:: get_covariance() -> float
   .. py:method:: get_correlation() -> float
   .. py:method:: get_mean_x() -> float
   .. py:method:: get_mean_y() -> float
   .. py:method:: process_covariance_batch(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray
   .. py:method:: process_correlation_batch(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

   .. code-block:: python

      sc = rrc.SlidingCovariance(3)
      for x, y in [(1, 2), (2, 4), (3, 6)]:
          sc.update(x, y)
      sc.get_covariance(), sc.get_correlation()
      # (2.0, 1.0)

----

.. py:class:: SlidingMomentsPrefix(window_size)

   Stateless batch engine for variance, skewness, and kurtosis using prefix
   sums of raw moments.  Faster than :py:class:`SlidingMoments` but
   susceptible to catastrophic cancellation for data with large values and
   small variance.  Use when numerical precision is not critical.

   :param window_size: int

   .. py:method:: variance_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray
   .. py:method:: skewness_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray
   .. py:method:: kurtosis_batch(x: numpy.ndarray, min_periods: int = 0) -> numpy.ndarray

   .. code-block:: python

      x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

      rrc.SlidingMomentsPrefix(3).variance_batch(x)
      # [nan, 0.5, 1., 1., 1.]

      rrc.SlidingMomentsPrefix(3).skewness_batch(x)
      # [nan, nan, 0., 0., 0.]

      rrc.SlidingMomentsPrefix(4).kurtosis_batch(x)
      # [nan, nan, nan, -1.2, -1.2]
