"""Tests for rolling_cov and rolling_cor high-level API."""
import numpy as np
import pandas as pd
import pytest

import robustrolling as rr

from conftest import nan_allclose


# ── Output type ────────────────────────────────────────────────────────────────

class TestCovCorOutputType:

    def test_series_input_returns_series_with_name_of_x(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="a")
        t = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], name="b")
        out = rr.rolling_cov(s, t, 3)
        assert isinstance(out, pd.Series)
        assert out.name == "a"

    def test_ndarray_input_returns_ndarray(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert isinstance(rr.rolling_cov(x, y, 3), np.ndarray)
        assert isinstance(rr.rolling_cor(x, y, 3), np.ndarray)

    def test_output_length_equals_input_length(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert len(rr.rolling_cov(x, y, 3)) == 5
        assert len(rr.rolling_cor(x, y, 3)) == 5

    def test_empty_input_returns_empty(self):
        out = rr.rolling_cov(np.array([]), np.array([]), 3)
        assert len(out) == 0


# ── Known values ───────────────────────────────────────────────────────────────

class TestCovCorKnownValues:

    def test_rolling_cov_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x * 2.0   # cov(2x, x) = 2 * var(x) = 2 * 1.0 = 2.0
        out = rr.rolling_cov(x, y, 3)
        nan_allclose(out, [np.nan, np.nan, 2.0, 2.0, 2.0])

    def test_rolling_cor_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rr.rolling_cor(x, x * 3.0, 3)
        np.testing.assert_allclose(out[2:], 1.0, atol=1e-12)

    def test_rolling_cor_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rr.rolling_cor(x, -x + 6.0, 3)
        np.testing.assert_allclose(out[2:], -1.0, atol=1e-12)

    def test_rolling_cov_default_min_periods_masks_warmup(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([4.0, 3.0, 2.0, 1.0])
        out = rr.rolling_cov(x, y, 3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert not np.isnan(out[2])


# ── Pandas comparison ──────────────────────────────────────────────────────────

class TestCovCorPandasComparison:

    def test_rolling_cov_matches_pandas_clean(self):
        np.random.seed(7)
        x = np.random.randn(200)
        y = np.random.randn(200)
        k = 8
        expected = pd.Series(x).rolling(k, min_periods=k).cov(pd.Series(y)).to_numpy()
        nan_allclose(rr.rolling_cov(x, y, k), expected, rtol=1e-10, atol=1e-10)

    def test_rolling_cor_matches_pandas_clean(self):
        np.random.seed(7)
        x = np.random.randn(200)
        y = np.random.randn(200)
        k = 8
        expected = pd.Series(x).rolling(k, min_periods=k).corr(pd.Series(y)).to_numpy()
        nan_allclose(rr.rolling_cor(x, y, k), expected, rtol=1e-10, atol=1e-10)

    def test_rolling_cov_matches_pandas_with_nan(self):
        np.random.seed(42)
        x = np.random.randn(300)
        y = np.random.randn(300)
        x[np.random.rand(300) < 0.15] = np.nan
        y[np.random.rand(300) < 0.15] = np.nan
        k, mp = 7, 3
        expected = pd.Series(x).rolling(k, min_periods=mp).cov(pd.Series(y)).to_numpy()
        nan_allclose(rr.rolling_cov(x, y, k, min_periods=mp), expected, rtol=1e-9, atol=1e-9)

    def test_rolling_cor_matches_pandas_with_nan(self):
        np.random.seed(42)
        x = np.random.randn(300)
        y = np.random.randn(300)
        x[np.random.rand(300) < 0.15] = np.nan
        y[np.random.rand(300) < 0.15] = np.nan
        k, mp = 7, 3
        expected = pd.Series(x).rolling(k, min_periods=mp).corr(pd.Series(y)).to_numpy()
        nan_allclose(rr.rolling_cor(x, y, k, min_periods=mp), expected, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("mp", [1, 2, 3])
    def test_rolling_cov_min_periods_matches_pandas(self, mp):
        np.random.seed(10 + mp)
        x = np.random.randn(100)
        y = np.random.randn(100)
        x[np.random.rand(100) < 0.2] = np.nan
        k = 6
        expected = pd.Series(x).rolling(k, min_periods=mp).cov(pd.Series(y)).to_numpy()
        nan_allclose(rr.rolling_cov(x, y, k, min_periods=mp), expected, rtol=1e-9, atol=1e-9)
