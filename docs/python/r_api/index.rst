R API Reference
===============

All functions accept a numeric vector ``x`` (and ``y`` for bivariate
functions), a ``window_size`` integer, and an optional ``min_periods``
parameter compatible with *pandas* semantics.

.. function:: rolling_cor(x, y, window_size, min_periods = window_size)

   *Rolling Correlation* — Computes the rolling Pearson correlation between two numeric vectors.

   :param x: A numeric vector of type double.
   :param y: A numeric vector of type double, same length as ``x``.
   :param window_size: Positive integer window length.
   :param min_periods: Minimum number of valid (non-``NA``) pairs required. Defaults to ``window_size``.
   :returns: A numeric vector with rolling correlation values.

   .. rubric:: Example

   .. code-block:: r

      x <- as.double(c(1, 2, 3, 4, 5))
      y <- as.double(c(2, 4, 6, 8, 10))
      rolling_cor(x, y, 3L)


----

.. function:: rolling_cov(x, y, window_size, min_periods = window_size)

   *Rolling Covariance* — Computes the rolling sample covariance (ddof=1) between two numeric vectors.

   :param x: A numeric vector of type double.
   :param y: A numeric vector of type double, same length as ``x``.
   :param window_size: Positive integer window length.
   :param min_periods: Minimum number of valid (non-``NA``) pairs required. Defaults to ``window_size``.
   :returns: A numeric vector with rolling covariance values.

   .. rubric:: Example

   .. code-block:: r

      x <- as.double(c(1, 2, 3, 4, 5))
      y <- as.double(c(2, 4, 6, 8, 10))
      rolling_cov(x, y, 3L)


----

.. function:: rolling_kurtosis(x, window_size, min_periods = window_size, method = "stable")

   *Rolling Kurtosis* — Computes the rolling excess kurtosis (Fisher) over a numeric vector.
   Requires at least 4 non-``NA`` observations per window.

   :param x: A numeric vector of type double.
   :param window_size: Positive integer window length.
   :param min_periods: Minimum number of non-``NA`` observations required in a window to return a result. Defaults to ``window_size``.
   :param method: ``"stable"`` (default) uses Terriberry's online algorithm. ``"fast"`` uses a prefix-sum approach (faster, but susceptible to catastrophic cancellation when values are large and variance is small).
   :returns: A numeric vector with rolling excess kurtosis values.

   .. rubric:: Example

   .. code-block:: r

      x <- as.double(c(1, 2, 3, 4, 5))
      rolling_kurtosis(x, 4L)


----

.. function:: rolling_max(x, window_size, min_periods = window_size)

   *Rolling Maximum* — Computes the rolling maximum over a numeric vector using a monotonic deque.

   :param x: A numeric vector of type double.
   :param window_size: Positive integer window length.
   :param min_periods: Minimum number of non-``NA`` observations required in a window to return a result. Defaults to ``window_size``.
   :returns: A numeric vector with rolling maximum values.

   .. rubric:: Example

   .. code-block:: r

      x <- as.double(c(1, 3, 2, 5, 4))
      rolling_max(x, 3L)


----

.. function:: rolling_mean(x, window_size, min_periods = window_size, assume_finite = FALSE)

   *Rolling Mean* — Computes the rolling mean over a numeric vector.

   :param x: A numeric vector of type double.
   :param window_size: Positive integer window length.
   :param min_periods: Minimum number of non-``NA`` observations required in a window to return a result. Defaults to ``window_size``.
   :param assume_finite: If ``TRUE``, assumes the input contains no ``NA`` values and uses a faster SIMD prefix-sum path. Passing ``TRUE`` when ``NA``s are present produces incorrect results. Defaults to ``FALSE``.
   :returns: A numeric vector with rolling mean values.

   .. rubric:: Example

   .. code-block:: r

      x <- as.double(c(1, 2, 3, 4))
      rolling_mean(x, 3L)


----

.. function:: rolling_median(x, window_size, min_periods = window_size)

   *Rolling Median* — Computes the rolling median over a numeric vector using an ordered multiset
   with a tracked median iterator. Time complexity: O(log n) per element.

   :param x: A numeric vector of type double.
   :param window_size: Positive integer window length.
   :param min_periods: Minimum number of non-``NA`` observations required in a window to return a result. Defaults to ``window_size``.
   :returns: A numeric vector with rolling median values.

   .. rubric:: Example

   .. code-block:: r

      x <- as.double(c(1, 3, 2, 5, 4))
      rolling_median(x, 3L)


----

.. function:: rolling_min(x, window_size, min_periods = window_size)

   *Rolling Minimum* — Computes the rolling minimum over a numeric vector using a monotonic deque.

   :param x: A numeric vector of type double.
   :param window_size: Positive integer window length.
   :param min_periods: Minimum number of non-``NA`` observations required in a window to return a result. Defaults to ``window_size``.
   :returns: A numeric vector with rolling minimum values.

   .. rubric:: Example

   .. code-block:: r

      x <- as.double(c(1, 3, 2, 5, 4))
      rolling_min(x, 3L)


----

.. function:: rolling_skewness(x, window_size, min_periods = window_size, method = "stable")

   *Rolling Skewness* — Computes the rolling adjusted Fisher-Pearson skewness over a numeric vector.
   Requires at least 3 non-``NA`` observations per window.

   :param x: A numeric vector of type double.
   :param window_size: Positive integer window length.
   :param min_periods: Minimum number of non-``NA`` observations required in a window to return a result. Defaults to ``window_size``.
   :param method: ``"stable"`` (default) uses Terriberry's online algorithm. ``"fast"`` uses a prefix-sum approach (faster, but susceptible to catastrophic cancellation when values are large and variance is small).
   :returns: A numeric vector with rolling skewness values.

   .. rubric:: Example

   .. code-block:: r

      x <- as.double(c(1, 2, 3, 4, 5))
      rolling_skewness(x, 3L)


----

.. function:: rolling_variance(x, window_size, min_periods = window_size, method = "stable")

   *Rolling Sample Variance* — Computes the rolling sample variance over a numeric vector.

   :param x: A numeric vector of type double.
   :param window_size: Positive integer window length.
   :param min_periods: Minimum number of non-``NA`` observations required in a window to return a result. Defaults to ``window_size`` (pandas semantics). Positions with fewer non-``NA`` values yield ``NA``.
   :param method: ``"stable"`` (default) uses the Welford online algorithm. ``"fast"`` uses a prefix-sum approach (faster, but susceptible to catastrophic cancellation when values are large and variance is small).
   :returns: A numeric vector with rolling sample variance values. Entries are ``NA`` when fewer than ``min_periods`` non-``NA`` observations are present in the window, and ``NaN`` when variance is undefined (fewer than two values).

   .. rubric:: Example

   .. code-block:: r

      x <- as.double(c(1, 2, 3, 4))
      rolling_variance(x, 3L)

