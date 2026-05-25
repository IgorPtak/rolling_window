"""Tests for the high-level API: rolling_max, min, median, variance, mean."""
import numpy as np
import pandas as pd
import pytest

import robustrolling as rr

from conftest import nan_allclose

_FNS = [rr.rolling_max, rr.rolling_min, rr.rolling_median, rr.rolling_variance, rr.rolling_mean]
_FN_IDS = ["max", "min", "median", "variance", "mean"]


# ── Output type preservation ───────────────────────────────────────────────────

class TestOutputType:

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_series_input_returns_series(self, fn):
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0], name="x")
        out = fn(s, 3)
        assert isinstance(out, pd.Series)
        assert out.name == "x"

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_series_preserves_datetime_index(self, fn):
        idx = pd.date_range("2024-01-01", periods=5)
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0], index=idx)
        out = fn(s, 3)
        assert isinstance(out, pd.Series)
        assert out.index.equals(idx)

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_series_preserves_range_index(self, fn):
        idx = pd.RangeIndex(start=10, stop=15)
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0], index=idx)
        assert fn(s, 3).index.equals(idx)

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_ndarray_input_returns_ndarray(self, fn):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        assert isinstance(fn(arr, 3), np.ndarray)

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_output_length_equals_input_length(self, fn):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        assert len(fn(arr, 3)) == len(arr)

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_output_dtype_is_float64(self, fn):
        arr = np.array([1, 3, 2, 5, 4], dtype=np.int32)
        assert fn(arr, 3).dtype == np.float64


# ── Pandas comparison ──────────────────────────────────────────────────────────

_PANDAS_CASES = [
    # label,                          data,                                  k   mp
    ("clean_default",   [1.0, 3.0, 2.0, 5.0, 4.0],                           3,  None),
    ("clean_mp1",       [1.0, 3.0, 2.0, 5.0, 4.0],                           3,  1),
    ("clean_mp2",       [1.0, 3.0, 2.0, 5.0, 4.0],                           3,  2),
    ("longer_default",  [-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0],               4,  None),
    ("longer_mp2",      [-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0],               4,  2),
    ("nan_mp1",         [1.0, np.nan, 3.0, 4.0, 5.0],                        3,  1),
    ("nan_mp2",         [1.0, np.nan, 3.0, 4.0, 5.0],                        3,  2),
    ("nan_default",     [1.0, np.nan, 3.0, 4.0, 5.0],                        3,  None),
    ("leading_nans",    [np.nan, np.nan, 3.0, 4.0],                          2,  1),
    ("k_gt_n",          [1.0, 2.0, 3.0],                                     10, 1),
    ("constant",        [5.0] * 6,                                           3,  None),
    ("negatives_mp1",   [-5.0, -1.0, -3.0, -2.0, -4.0],                      3,  1),
    ("mixed_nan_mp1",   [np.nan, 1.0, np.nan, 3.0, np.nan],                  2,  1),
    ("window2_nan_gap", [5.0, 1.0, np.nan, 0.0],                             2,  1),
]

_CASE_IDS = [c[0] for c in _PANDAS_CASES]
_CASE_PARAMS = [(c[1], c[2], c[3]) for c in _PANDAS_CASES]


def _pd_mp(mp, k):
    return k if mp is None else mp


def _kw(mp):
    return {} if mp is None else {"min_periods": mp}


class TestPandasComparison:

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_max_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).max().to_numpy()
        nan_allclose(rr.rolling_max(arr, k, **_kw(mp)), expected)

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_min_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).min().to_numpy()
        nan_allclose(rr.rolling_min(arr, k, **_kw(mp)), expected)

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_median_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).median().to_numpy()
        nan_allclose(rr.rolling_median(arr, k, **_kw(mp)), expected)

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_variance_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).var().to_numpy()
        nan_allclose(rr.rolling_variance(arr, k, **_kw(mp)), expected)

    @pytest.mark.parametrize("data,k,mp", _CASE_PARAMS, ids=_CASE_IDS)
    def test_rolling_mean_matches_pandas(self, data, k, mp):
        arr = np.array(data, dtype=np.float64)
        expected = pd.Series(arr).rolling(k, min_periods=_pd_mp(mp, k)).mean().to_numpy()
        nan_allclose(rr.rolling_mean(arr, k, **_kw(mp)), expected)


# ── min_periods edge cases ─────────────────────────────────────────────────────

class TestMinPeriods:

    def test_default_equals_window_size_max_min_median(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        for fn in (rr.rolling_max, rr.rolling_min, rr.rolling_median):
            out = fn(x, 3)
            assert np.isnan(out[0]) and np.isnan(out[1]) and not np.isnan(out[2])

    def test_default_equals_window_size_variance(self):
        out = rr.rolling_variance(np.array([1.0, 2.0, 3.0, 4.0]), 3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert pytest.approx(out[2], abs=1e-10) == 1.0

    def test_min_periods_0_no_masking_on_clean_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        for fn in (rr.rolling_max, rr.rolling_min, rr.rolling_median):
            assert not np.any(np.isnan(fn(x, 3, min_periods=0)))

    def test_nan_series_default_all_masked(self):
        x = np.array([1.0, np.nan, 3.0, 4.0])
        for fn in (rr.rolling_max, rr.rolling_min, rr.rolling_median):
            assert np.all(np.isnan(fn(x, 3)))

    def test_window_expiry_max_nan_gap(self):
        x = np.array([5.0, 1.0, np.nan, 0.0])
        out = rr.rolling_max(x, 2, min_periods=1)
        assert pytest.approx(out[2], abs=1e-12) == 1.0
        assert pytest.approx(out[3], abs=1e-12) == 0.0

    def test_window_expiry_min_nan_gap(self):
        x = np.array([1.0, 5.0, np.nan, 9.0])
        out = rr.rolling_min(x, 2, min_periods=1)
        assert pytest.approx(out[2], abs=1e-12) == 5.0
        assert pytest.approx(out[3], abs=1e-12) == 9.0

    def test_variance_min_periods_1_single_value_is_nan(self):
        out = rr.rolling_variance(np.array([1.0, 5.0]), 2, min_periods=1)
        assert np.isnan(out[0])
        assert pytest.approx(out[1], abs=1e-10) == 8.0

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_validation_negative_min_periods(self, fn):
        with pytest.raises(Exception):
            fn(np.array([1.0, 2.0, 3.0]), 3, min_periods=-1)

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_validation_min_periods_exceeds_window(self, fn):
        with pytest.raises(Exception):
            fn(np.array([1.0, 2.0, 3.0]), 3, min_periods=4)

    @pytest.mark.parametrize("fn", _FNS, ids=_FN_IDS)
    def test_empty_input_returns_empty(self, fn):
        assert len(fn(np.array([]), 3)) == 0
