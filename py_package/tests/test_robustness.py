"""Memory layout, dtype, and fuzz tests — cross-cutting robustness."""
import numpy as np
import pandas as pd
import pytest

import robust_rolling_core as rrc
import robustrolling as rr

from conftest import nan_allclose

_ENGINE_CLASSES = [rrc.MonotonicMax, rrc.MonotonicMin, rrc.MultisetMedian, rrc.SlidingWelford]
_ENGINE_IDS = ["MonotonicMax", "MonotonicMin", "MultisetMedian", "SlidingWelford"]

_SIMPLE_FNS = [rr.rolling_max, rr.rolling_min, rr.rolling_median, rr.rolling_variance, rr.rolling_mean]
_SIMPLE_IDS = ["max", "min", "median", "variance", "mean"]


# ── Memory layout and dtype ────────────────────────────────────────────────────

class TestMemoryLayout:

    @pytest.mark.parametrize("cls", _ENGINE_CLASSES, ids=_ENGINE_IDS)
    def test_strided_every_other_element(self, cls):
        x = np.arange(10, dtype=np.float64)
        strided = x[::2]
        assert not strided.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(
            cls(2).process_batch(strided),
            cls(2).process_batch(np.ascontiguousarray(strided))
        )

    @pytest.mark.parametrize("cls", _ENGINE_CLASSES, ids=_ENGINE_IDS)
    def test_strided_column_slice(self, cls):
        col = np.arange(16.0).reshape(8, 2)[:, 0]
        assert not col.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(
            cls(3).process_batch(col),
            cls(3).process_batch(np.ascontiguousarray(col))
        )

    @pytest.mark.parametrize("cls", _ENGINE_CLASSES, ids=_ENGINE_IDS)
    def test_strided_reversed(self, cls):
        rev = np.arange(8, dtype=np.float64)[::-1]
        assert not rev.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(
            cls(2).process_batch(rev),
            cls(2).process_batch(np.ascontiguousarray(rev))
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.int32, np.int64],
                             ids=["float32", "int32", "int64"])
    def test_numeric_dtype_cast_in_engine(self, dtype):
        x = np.array([1, 3, 2, 5, 4], dtype=dtype)
        out = rrc.MonotonicMax(3).process_batch(x)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, rrc.MonotonicMax(3).process_batch(x.astype(np.float64)))

    @pytest.mark.parametrize("fn", _SIMPLE_FNS, ids=_SIMPLE_IDS)
    @pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_],
                             ids=["float32", "int64", "bool"])
    def test_numeric_dtype_in_high_level_api(self, fn, dtype):
        x = np.array([1, 3, 2, 5, 4], dtype=dtype)
        out = fn(x, 3)
        assert out.dtype == np.float64
        nan_allclose(out, fn(x.astype(np.float64), 3))

    @pytest.mark.parametrize("fn", _SIMPLE_FNS, ids=_SIMPLE_IDS)
    def test_strided_high_level_api(self, fn):
        base = np.array([1.0, 3.0, 2.0, 7.0, 4.0, 6.0, 5.0, 8.0])
        strided = base[::2]
        assert not strided.flags["C_CONTIGUOUS"]
        nan_allclose(np.asarray(fn(strided, 2)),
                     np.asarray(fn(np.ascontiguousarray(strided), 2)))


# ── Fuzz / stress ──────────────────────────────────────────────────────────────

class TestFuzz:

    @pytest.mark.parametrize("fn,pd_fn", [
        (rr.rolling_max,      lambda s, k, mp: s.rolling(k, min_periods=mp).max().to_numpy()),
        (rr.rolling_min,      lambda s, k, mp: s.rolling(k, min_periods=mp).min().to_numpy()),
        (rr.rolling_median,   lambda s, k, mp: s.rolling(k, min_periods=mp).median().to_numpy()),
        (rr.rolling_variance, lambda s, k, mp: s.rolling(k, min_periods=mp).var().to_numpy()),
        (rr.rolling_mean,     lambda s, k, mp: s.rolling(k, min_periods=mp).mean().to_numpy()),
    ], ids=["max", "min", "median", "variance", "mean"])
    def test_random_nan_20pct_matches_pandas(self, fn, pd_fn):
        np.random.seed(42)
        x = np.random.randn(5000)
        x[np.random.rand(5000) < 0.2] = np.nan
        k, mp = 15, 5
        nan_allclose(fn(x, k, min_periods=mp), pd_fn(pd.Series(x), k, mp),
                     rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("fn,pd_fn", [
        (lambda x, k, mp: rr.rolling_skewness(x, k, min_periods=mp),
         lambda s, k, mp: s.rolling(k, min_periods=mp).skew().to_numpy()),
        (lambda x, k, mp: rr.rolling_kurtosis(x, k, min_periods=mp),
         lambda s, k, mp: s.rolling(k, min_periods=mp).kurt().to_numpy()),
    ], ids=["skewness", "kurtosis"])
    def test_moments_random_nan_matches_pandas(self, fn, pd_fn):
        np.random.seed(99)
        x = np.random.randn(2000)
        x[np.random.rand(2000) < 0.15] = np.nan
        k, mp = 10, 4
        nan_allclose(fn(x, k, mp), pd_fn(pd.Series(x), k, mp), rtol=1e-8, atol=1e-8)

    def test_stress_many_instances_no_crash(self):
        x = np.random.randn(100).astype(np.float64)
        for _ in range(2000):
            rrc.MultisetMedian(10).process_batch(x)
