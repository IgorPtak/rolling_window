"""Tests for rolling_skewness, rolling_kurtosis and method='fast'."""
import numpy as np
import pandas as pd
import pytest

import robustrolling as rr

from conftest import nan_allclose

_FNS = [rr.rolling_skewness, rr.rolling_kurtosis]
_FN_IDS = ["skewness", "kurtosis"]

# Minimum valid observations per metric (algorithm threshold, not min_periods)
_MIN_OBS = {rr.rolling_skewness: 3, rr.rolling_kurtosis: 4}


# ── Output type preservation ───────────────────────────────────────────────────

class TestMomentsOutputType:

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_series_input_returns_series(self, fn):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
        out = fn(s, 5)
        assert isinstance(out, pd.Series)
        assert out.name == "x"

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_series_preserves_index(self, fn):
        idx = pd.date_range("2024-01-01", periods=5)
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        assert fn(s, 5).index.equals(idx)

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_ndarray_input_returns_ndarray(self, fn):
        assert isinstance(fn(np.arange(1.0, 8.0), 5), np.ndarray)

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_output_length_equals_input_length(self, fn):
        arr = np.arange(1.0, 8.0)
        assert len(fn(arr, 5)) == len(arr)

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_output_dtype_is_float64(self, fn):
        arr = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        assert fn(arr, 5).dtype == np.float64

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_empty_input_returns_empty(self, fn):
        assert len(fn(np.array([]), 5)) == 0


# ── Pandas comparison (stable method) ─────────────────────────────────────────

_MOMENT_CASES = [
    # label,           data,                                      k
    ("basic",          [1.0, 3.0, -1.0, 5.0, 2.0, 8.0, 0.0],      4),
    ("longer",         [-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0,
                         3.0, 5.0, 2.0],                          5),
    ("with_nan",       [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],    4),
    ("leading_nan",    [np.nan, np.nan, 1.0, 2.0, 3.0, 4.0],      4),
    ("negatives",      [-5.0, -1.0, -3.0, -2.0, -4.0, -6.0],      4),
]

_MOMENT_IDS = [c[0] for c in _MOMENT_CASES]
_MOMENT_PARAMS = [(c[1], c[2]) for c in _MOMENT_CASES]


class TestMomentsPandasComparison:

    @pytest.mark.parametrize("data,k", _MOMENT_PARAMS, ids=_MOMENT_IDS)
    def test_rolling_skewness_matches_pandas(self, data, k):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=3).skew().to_numpy()
        nan_allclose(rr.rolling_skewness(arr, k, min_periods=3), expected, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("data,k", _MOMENT_PARAMS, ids=_MOMENT_IDS)
    def test_rolling_kurtosis_matches_pandas(self, data, k):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=4).kurt().to_numpy()
        nan_allclose(rr.rolling_kurtosis(arr, k, min_periods=4), expected, rtol=1e-9, atol=1e-9)

    def test_skewness_symmetric_window_is_zero(self):
        # [1,2,3] is symmetric -> skewness = 0
        out = rr.rolling_skewness(np.array([1.0, 2.0, 3.0]), 3)
        assert pytest.approx(out[2], abs=1e-12) == 0.0

    def test_kurtosis_uniform_window_known_value(self):
        # [1,2,3,4]: excess kurtosis = -1.2
        out = rr.rolling_kurtosis(np.array([1.0, 2.0, 3.0, 4.0]), 4)
        assert pytest.approx(out[3], abs=1e-10) == -1.2

    def test_skewness_window_smaller_than_min_obs_always_nan(self):
        # window=2 -> never >=3 observations -> always NaN
        out = rr.rolling_skewness(np.arange(1.0, 8.0), 2)
        assert np.all(np.isnan(out))

    def test_kurtosis_window_smaller_than_min_obs_always_nan(self):
        # window=3 -> never >=4 observations -> always NaN
        out = rr.rolling_kurtosis(np.arange(1.0, 8.0), 3)
        assert np.all(np.isnan(out))


# ── method="fast" ──────────────────────────────────────────────────────────────

class TestFastMethod:

    @pytest.mark.parametrize("fn,pd_fn", [
        (lambda x, k: rr.rolling_variance(x, k, method="fast"),
         lambda s, k: s.rolling(k).var().to_numpy()),
        (lambda x, k: rr.rolling_skewness(x, k, method="fast"),
         lambda s, k: s.rolling(k).skew().to_numpy()),
        (lambda x, k: rr.rolling_kurtosis(x, k, method="fast"),
         lambda s, k: s.rolling(k).kurt().to_numpy()),
    ], ids=["variance", "skewness", "kurtosis"])
    def test_fast_matches_pandas_no_nan(self, fn, pd_fn):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(300)
        k = 10
        nan_allclose(fn(x, k), pd_fn(pd.Series(x), k), rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("fn_stable,fn_fast", [
        (lambda x, k, mp: rr.rolling_variance(x, k, min_periods=mp),
         lambda x, k, mp: rr.rolling_variance(x, k, min_periods=mp, method="fast")),
        (lambda x, k, mp: rr.rolling_skewness(x, k, min_periods=mp),
         lambda x, k, mp: rr.rolling_skewness(x, k, min_periods=mp, method="fast")),
        (lambda x, k, mp: rr.rolling_kurtosis(x, k, min_periods=mp),
         lambda x, k, mp: rr.rolling_kurtosis(x, k, min_periods=mp, method="fast")),
    ], ids=["variance", "skewness", "kurtosis"])
    def test_fast_agrees_with_stable_with_nan(self, fn_stable, fn_fast):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(300)
        x[rng.random(300) < 0.1] = np.nan
        k, mp = 10, 5
        nan_allclose(fn_fast(x, k, mp), fn_stable(x, k, mp), rtol=1e-9, atol=1e-9)

    def test_fast_variance_warmup_nan(self):
        x = np.arange(1.0, 11.0)
        out = rr.rolling_variance(x, 5, method="fast")
        assert all(np.isnan(out[:4]))
        assert not np.isnan(out[4])

    def test_fast_skewness_needs_3_obs(self):
        out = rr.rolling_skewness(np.arange(1.0, 11.0), 5, method="fast")
        assert all(np.isnan(out[:4]))

    def test_fast_kurtosis_needs_4_obs(self):
        out = rr.rolling_kurtosis(np.arange(1.0, 11.0), 5, method="fast")
        assert all(np.isnan(out[:4]))

    def test_fast_min_periods(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rr.rolling_variance(x, 3, min_periods=2, method="fast")
        assert np.isnan(out[0])
        assert not np.isnan(out[1])

    def test_fast_all_nan_window_returns_nan(self):
        x = np.array([np.nan, np.nan, np.nan, 1.0, 2.0, 3.0])
        out = rr.rolling_variance(x, 3, min_periods=1, method="fast")
        assert np.isnan(out[2])
        assert not np.isnan(out[5])

    @pytest.mark.parametrize("fn", [rr.rolling_variance, rr.rolling_skewness, rr.rolling_kurtosis],
                             ids=["variance", "skewness", "kurtosis"])
    def test_fast_empty_input(self, fn):
        assert len(fn(np.array([]), 3, method="fast")) == 0
