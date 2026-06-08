from __future__ import annotations

import numpy as np

from robust_rolling_core import (
    FlatMedian,
    MonotonicMax,
    MonotonicMin,
    MultisetMedian,
    SlidingCovariance,
    SlidingMean,
    SlidingMedian,
    SlidingMoments,
    SlidingMomentsPrefix,
    SlidingWelford,
    TwoHeapMedian,
)

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

__all__ = [
    "FlatMedian",
    "MonotonicMax",
    "MonotonicMin",
    "MultisetMedian",
    "SlidingCovariance",
    "SlidingMedian",
    "SlidingMoments",
    "SlidingWelford",
    "TwoHeapMedian",
    "rolling_max",
    "rolling_min",
    "rolling_variance",
    "rolling_median",
    "rolling_mean",
    "rolling_skewness",
    "rolling_kurtosis",
    "rolling_cov",
    "rolling_cor",
]


def _to_float64(x) -> np.ndarray:
    if _HAS_PANDAS and isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _wrap(result: np.ndarray, original):
    if _HAS_PANDAS and isinstance(original, pd.Series):
        return pd.Series(result, index=original.index, name=original.name)
    return result


def _resolve_min_periods(min_periods: int | None, window_size: int) -> int:
    mp = window_size if min_periods is None else int(min_periods)
    if mp < 0 or mp > window_size:
        raise ValueError(
            f"min_periods must be in [0, window_size], got {mp}"
        )
    return mp


def rolling_max(x, window_size: int, min_periods: int | None = None):
    """
    Compute the rolling maximum over a sliding window.

    Parameters
    ----------
    x : array-like
        Input sequence. Accepts ``np.ndarray`` and ``pd.Series``.
    window_size : int
        Number of observations in the sliding window.
    min_periods : int, optional
        Minimum number of non-NaN observations required to return a result.
        Defaults to ``window_size`` (pandas-compatible semantics).

    Returns
    -------
    numpy.ndarray or pandas.Series
        Rolling maximum values. Positions with fewer than ``min_periods``
        valid observations are ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> import robustrolling as rr
    >>> x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    >>> rr.rolling_max(x, 3)
    array([nan, nan,  3.,  5.,  5.])
    """
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = MonotonicMax(window_size).process_batch(arr, mp)
    return _wrap(result, x)


def rolling_min(x, window_size: int, min_periods: int | None = None):
    """
    Compute the rolling minimum over a sliding window.

    Parameters
    ----------
    x : array-like
        Input sequence. Accepts ``np.ndarray`` and ``pd.Series``.
    window_size : int
        Number of observations in the sliding window.
    min_periods : int, optional
        Minimum number of non-NaN observations required to return a result.
        Defaults to ``window_size`` (pandas-compatible semantics).

    Returns
    -------
    numpy.ndarray or pandas.Series
        Rolling minimum values. Positions with fewer than ``min_periods``
        valid observations are ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> import robustrolling as rr
    >>> x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    >>> rr.rolling_min(x, 3)
    array([nan, nan,  1.,  2.,  2.])
    """
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = MonotonicMin(window_size).process_batch(arr, mp)
    return _wrap(result, x)


def rolling_variance(x, window_size: int, min_periods: int | None = None,
                     method: str = "stable"):
    """
    Compute the rolling sample variance (ddof=1) over a sliding window.

    Parameters
    ----------
    x : array-like
        Input sequence. Accepts ``np.ndarray`` and ``pd.Series``.
    window_size : int
        Number of observations in the sliding window.
    min_periods : int, optional
        Minimum number of non-NaN observations required to return a result.
        Defaults to ``window_size`` (pandas-compatible semantics).
    method : {"stable", "fast"}, optional
        ``"stable"`` uses the Welford online algorithm (numerically stable,
        default). ``"fast"`` uses a prefix-sum approach (faster for large
        arrays, but susceptible to catastrophic cancellation when values are
        large and variance is small).

    Returns
    -------
    numpy.ndarray or pandas.Series
        Rolling sample variance. Returns ``nan`` when fewer than
        ``min_periods`` valid observations are present, or when fewer than
        two observations are available (variance undefined).

    Examples
    --------
    >>> import numpy as np
    >>> import robustrolling as rr
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> rr.rolling_variance(x, 3)
    array([nan, nan,  1.,  1.,  1.])
    """
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    if method == "fast":
        result = SlidingMomentsPrefix(window_size).variance_batch(arr, mp)
    else:
        result = SlidingWelford(window_size).process_batch(arr, mp)
    return _wrap(result, x)


def rolling_median(x, window_size: int, min_periods: int | None = None,
                   expect_nan: bool = False):
    """
    Compute the rolling median over a sliding window.

    Automatically selects the fastest algorithm for the given window size and
    NaN density. Use ``expect_nan=True`` if the input is expected to contain
    a significant fraction of NaN values (avoids MultisetMedian's O(n)
    iterator-shift degradation at large windows).

    Parameters
    ----------
    x : array-like
        Input sequence. Accepts ``np.ndarray`` and ``pd.Series``.
    window_size : int
        Number of observations in the sliding window.
    min_periods : int, optional
        Minimum number of non-NaN observations required to return a result.
        Defaults to ``window_size`` (pandas-compatible semantics).
    expect_nan : bool, optional
        Hint that the input may contain many NaN values. Switches the
        dispatch thresholds to NaN-robust paths. Defaults to ``False``.

    Returns
    -------
    numpy.ndarray or pandas.Series
        Rolling median values. Positions with fewer than ``min_periods``
        valid observations are ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> import robustrolling as rr
    >>> x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    >>> rr.rolling_median(x, 3)
    array([nan, nan,  2.,  3.,  4.])
    """
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingMedian(window_size, expect_nan).process_batch(arr, mp)
    return _wrap(result, x)


def rolling_mean(x, window_size: int, min_periods: int | None = None, assume_finite: bool = False):
    """
    Compute the rolling arithmetic mean over a sliding window.

    Parameters
    ----------
    x : array-like
        Input sequence. Accepts ``np.ndarray`` and ``pd.Series``.
    window_size : int
        Number of observations in the sliding window.
    min_periods : int, optional
        Minimum number of non-NaN observations required to return a result.
        Defaults to ``window_size`` (pandas-compatible semantics).
    assume_finite : bool, optional
        If ``True``, assumes the input contains no ``NaN`` values and uses a
        faster single-pass prefix-sum path with SIMD acceleration.
        Passing ``True`` when the input does contain ``NaN`` produces
        incorrect results. Defaults to ``False``.
    Returns
    -------
    numpy.ndarray or pandas.Series
        Rolling mean values. Positions with fewer than ``min_periods``
        valid observations are ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> import robustrolling as rr
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> rr.rolling_mean(x, 3)
    array([nan, nan,  2.,  3.,  4.])
    """
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingMean(window_size).process_batch(arr, mp, assume_finite)
    return _wrap(result, x)


def rolling_skewness(x, window_size: int, min_periods: int | None = None,
                     method: str = "stable"):
    """
    Compute the rolling adjusted Fisher-Pearson skewness over a sliding window.

    Parameters
    ----------
    x : array-like
        Input sequence. Accepts ``np.ndarray`` and ``pd.Series``.
    window_size : int
        Number of observations in the sliding window.
    min_periods : int, optional
        Minimum number of non-NaN observations required to return a result.
        Defaults to ``window_size`` (pandas-compatible semantics).
    method : {"stable", "fast"}, optional
        ``"stable"`` uses Terriberry's online algorithm (numerically stable,
        default). ``"fast"`` uses a prefix-sum approach (faster for large
        arrays, but susceptible to catastrophic cancellation).

    Returns
    -------
    numpy.ndarray or pandas.Series
        Rolling skewness values. Returns ``nan`` when fewer than
        ``min_periods`` valid observations are present, or when fewer than
        three observations are available.

    Examples
    --------
    >>> import numpy as np
    >>> import robustrolling as rr
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> rr.rolling_skewness(x, 3)
    array([nan, nan,  0.,  0.,  0.])
    """
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    if method == "fast":
        result = SlidingMomentsPrefix(window_size).skewness_batch(arr, mp)
    else:
        result = SlidingMoments(window_size).process_skewness_batch(arr, mp)
    return _wrap(result, x)


def rolling_kurtosis(x, window_size: int, min_periods: int | None = None,
                     method: str = "stable"):
    """
    Compute the rolling excess kurtosis (Fisher definition) over a sliding window.

    Returns excess kurtosis (normal distribution = 0).
    Requires at least 4 valid observations per window.

    Parameters
    ----------
    x : array-like
        Input sequence. Accepts ``np.ndarray`` and ``pd.Series``.
    window_size : int
        Number of observations in the sliding window.
    min_periods : int, optional
        Minimum number of non-NaN observations required to return a result.
        Defaults to ``window_size`` (pandas-compatible semantics).
    method : {"stable", "fast"}, optional
        ``"stable"`` uses Terriberry's online algorithm (numerically stable,
        default). ``"fast"`` uses a prefix-sum approach (faster for large
        arrays, but susceptible to catastrophic cancellation).

    Returns
    -------
    numpy.ndarray or pandas.Series
        Rolling excess kurtosis values. Returns ``nan`` when fewer than
        ``min_periods`` valid observations are present, or when fewer than
        four observations are available.

    Examples
    --------
    >>> import numpy as np
    >>> import robustrolling as rr
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> rr.rolling_kurtosis(x, 4)
    array([nan, nan, nan, -1.2, -1.2])
    """
    arr = _to_float64(x)
    mp = _resolve_min_periods(min_periods, window_size)
    if method == "fast":
        result = SlidingMomentsPrefix(window_size).kurtosis_batch(arr, mp)
    else:
        result = SlidingMoments(window_size).process_kurtosis_batch(arr, mp)
    return _wrap(result, x)


def _count_valid_pairs_in_window(x: np.ndarray, y: np.ndarray,
                                  window_size: int) -> np.ndarray:
    both_valid = (~np.isnan(x) & ~np.isnan(y)).astype(np.float64)
    cum = np.cumsum(both_valid)
    lagged = np.empty_like(cum)
    lagged[:window_size] = 0.0
    if len(x) > window_size:
        lagged[window_size:] = cum[:-window_size]
    return cum - lagged


def _apply_min_periods_pair(result: np.ndarray, x: np.ndarray, y: np.ndarray,
                             window_size: int, min_periods: int) -> np.ndarray:
    if min_periods == 0 or len(x) == 0:
        return result
    valid_count = _count_valid_pairs_in_window(x, y, window_size)
    result = result.copy()
    result[valid_count < min_periods] = np.nan
    return result


def rolling_cov(x, y, window_size: int, min_periods: int | None = None):
    """
    Compute the rolling sample covariance (ddof=1) over a sliding window.

    Uses the 2D Welford online algorithm for O(1) updates.

    Parameters
    ----------
    x : array-like
        First input sequence. Accepts ``np.ndarray`` and ``pd.Series``.
    y : array-like
        Second input sequence, same length as ``x``.
    window_size : int
        Number of observations in the sliding window.
    min_periods : int, optional
        Minimum number of valid (non-NaN) pairs required to return a result.
        Defaults to ``window_size`` (pandas-compatible semantics).

    Returns
    -------
    numpy.ndarray or pandas.Series
        Rolling sample covariance values. Positions with fewer than
        ``min_periods`` valid pairs are ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> import robustrolling as rr
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> rr.rolling_cov(x, y, 3)
    array([nan, nan,  2.,  2.,  2.])
    """
    ax = _to_float64(x)
    ay = _to_float64(y)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingCovariance(window_size).process_covariance_batch(ax, ay)
    result = _apply_min_periods_pair(result, ax, ay, window_size, mp)
    return _wrap(result, x)


def rolling_cor(x, y, window_size: int, min_periods: int | None = None):
    """
    Compute the rolling Pearson correlation coefficient over a sliding window.

    Uses the 2D Welford online algorithm for O(1) updates.

    Parameters
    ----------
    x : array-like
        First input sequence. Accepts ``np.ndarray`` and ``pd.Series``.
    y : array-like
        Second input sequence, same length as ``x``.
    window_size : int
        Number of observations in the sliding window.
    min_periods : int, optional
        Minimum number of valid (non-NaN) pairs required to return a result.
        Defaults to ``window_size`` (pandas-compatible semantics).

    Returns
    -------
    numpy.ndarray or pandas.Series
        Rolling Pearson correlation values in [-1, 1]. Positions with fewer
        than ``min_periods`` valid pairs are ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> import robustrolling as rr
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> rr.rolling_cor(x, y, 3)
    array([nan, nan,  1.,  1.,  1.])
    """
    ax = _to_float64(x)
    ay = _to_float64(y)
    mp = _resolve_min_periods(min_periods, window_size)
    result = SlidingCovariance(window_size).process_correlation_batch(ax, ay)
    result = _apply_min_periods_pair(result, ax, ay, window_size, mp)
    return _wrap(result, x)
