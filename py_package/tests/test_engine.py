"""Tests for the C++ engine layer (rrc.*) — stateful rolling objects."""
import math

import numpy as np
import pandas as pd
import pytest

import robust_rolling_core as rrc

from conftest import nan_allclose, var_ref, mean_ref, median_ref, cov_ref, cor_ref


# ── SlidingWelford ─────────────────────────────────────────────────────────────

class TestSlidingWelford:

    def test_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rrc.SlidingWelford(3).process_batch(x)
        nan_allclose(out, [np.nan, 0.5, 1.0, 1.0, 1.0])

    def test_window_size_1_all_nan(self):
        out = rrc.SlidingWelford(1).process_batch(np.array([10.0, 11.0, 12.0]))
        assert np.all(np.isnan(out))

    def test_window_larger_than_array(self):
        out = rrc.SlidingWelford(10).process_batch(np.array([2.0, 4.0, 6.0]))
        assert np.isnan(out[0])
        assert np.isclose(out[1], 2.0, rtol=1e-12)   # var([2,4])
        assert np.isclose(out[2], 4.0, rtol=1e-12)   # var([2,4,6])

    def test_constant_zero_variance(self):
        out = rrc.SlidingWelford(4).process_batch(np.full(8, 5.0))
        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1:], 0.0, atol=1e-12)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 10.0, 7.0, 7.0, 8.0])
        out = rrc.SlidingWelford(k).process_batch(x)
        np.testing.assert_allclose(out, var_ref(x, k), rtol=1e-11, atol=1e-11,
                                   equal_nan=True)

    def test_nan_does_not_contribute(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = rrc.SlidingWelford(3).process_batch(x)
        assert np.isclose(out[2], 0.5, atol=1e-12)  # var([1,2]) = 0.5

    def test_empty_input(self):
        out = rrc.SlidingWelford(3).process_batch(np.array([]))
        assert len(out) == 0 and out.dtype == np.float64

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.SlidingWelford(0)

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.SlidingWelford(2).process_batch(np.ones((2, 3)))

    def test_integer_input_converted_to_float64(self):
        out = rrc.SlidingWelford(2).process_batch(np.array([1, 2, 3, 4], dtype=np.int32))
        assert out.dtype == np.float64


# ── MonotonicMax ───────────────────────────────────────────────────────────────

class TestMonotonicMax:

    def test_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 2.0, 5.0])
        out = rrc.MonotonicMax(3).process_batch(x)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 3.0, 5.0], atol=1e-12)

    def test_window_size_1_identity(self):
        x = np.array([-1.0, 0.0, 10.0, 2.0])
        np.testing.assert_allclose(rrc.MonotonicMax(1).process_batch(x), x)

    def test_window_larger_than_array_cumulative(self):
        x = np.array([2.0, -3.0, 7.0, 1.0])
        np.testing.assert_allclose(rrc.MonotonicMax(20).process_batch(x),
                                   np.maximum.accumulate(x))

    def test_decreasing_sequence(self):
        x = np.array([9.0, 7.0, 5.0, 3.0, 1.0])
        np.testing.assert_allclose(rrc.MonotonicMax(3).process_batch(x),
                                   [9.0, 9.0, 9.0, 7.0, 5.0])

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0])
        out = rrc.MonotonicMax(k).process_batch(x)
        expected = np.array([np.max(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_nan_advances_window_returns_current_max(self):
        x = np.array([1.0, 2.0, np.nan, 1.0])
        out = rrc.MonotonicMax(2).process_batch(x)
        assert out[2] == 2.0   # window=[2,NaN], max=2
        assert out[3] == 1.0   # window=[NaN,1], max=1

    def test_nan_at_start_empty_window_returns_nan(self):
        out = rrc.MonotonicMax(2).process_batch(np.array([np.nan, 2.0, 3.0]))
        assert np.isnan(out[0])

    def test_empty_input(self):
        out = rrc.MonotonicMax(3).process_batch(np.array([]))
        assert len(out) == 0 and out.dtype == np.float64

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.MonotonicMax(0)

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.MonotonicMax(2).process_batch(np.ones((2, 3)))

    def test_integer_input_converted_to_float64(self):
        out = rrc.MonotonicMax(2).process_batch(np.array([1, 2, 3], dtype=np.int32))
        assert out.dtype == np.float64


# ── MonotonicMin ───────────────────────────────────────────────────────────────

class TestMonotonicMin:

    def test_known_values(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        out = rrc.MonotonicMin(3).process_batch(x)
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 2.0, 2.0], atol=1e-12)

    def test_window_size_1_identity(self):
        x = np.array([-1.0, 0.0, 10.0, 2.0])
        np.testing.assert_allclose(rrc.MonotonicMin(1).process_batch(x), x)

    def test_window_larger_than_array_cumulative(self):
        x = np.array([2.0, -3.0, 7.0, 1.0])
        np.testing.assert_allclose(rrc.MonotonicMin(20).process_batch(x),
                                   np.minimum.accumulate(x))

    def test_increasing_sequence(self):
        x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        np.testing.assert_allclose(rrc.MonotonicMin(3).process_batch(x),
                                   [1.0, 1.0, 1.0, 3.0, 5.0])

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-2.0, 6.0, 1.0, 8.0, 0.0, 8.0, -1.0])
        out = rrc.MonotonicMin(k).process_batch(x)
        expected = np.array([np.min(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_nan_advances_window_returns_current_min(self):
        x = np.array([5.0, 1.0, np.nan, 9.0])
        out = rrc.MonotonicMin(2).process_batch(x)
        assert out[2] == 1.0   # window=[1,NaN], min=1
        assert out[3] == 9.0   # window=[NaN,9], min=9

    def test_nan_at_start_empty_window_returns_nan(self):
        out = rrc.MonotonicMin(2).process_batch(np.array([np.nan, 2.0, 3.0]))
        assert np.isnan(out[0])

    def test_empty_input(self):
        out = rrc.MonotonicMin(3).process_batch(np.array([]))
        assert len(out) == 0

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.MonotonicMin(0)

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.MonotonicMin(2).process_batch(np.ones((2, 3)))


# ── MultisetMedian ─────────────────────────────────────────────────────────────

class TestMultisetMedian:

    def test_known_values_odd_window(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        np.testing.assert_allclose(rrc.MultisetMedian(3).process_batch(x),
                                   median_ref(x, 3), rtol=1e-12)

    def test_known_values_even_window(self):
        x = np.array([1.0, 3.0, 2.0, 4.0])
        np.testing.assert_allclose(rrc.MultisetMedian(4).process_batch(x),
                                   median_ref(x, 4), rtol=1e-12)

    def test_window_size_1_identity(self):
        x = np.array([-1.0, 0.0, 10.0, 2.0])
        np.testing.assert_allclose(rrc.MultisetMedian(1).process_batch(x), x)

    def test_window_size_2_regression(self):
        # Regression: window_size=2 caused segfault before fix
        x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        np.testing.assert_allclose(rrc.MultisetMedian(2).process_batch(x),
                                   median_ref(x, 2), rtol=1e-12)

    def test_even_window_descending_fill_regression(self):
        # Regression: even window filled descending caused wrong mid_ position
        x = np.array([4.0, 3.0, 2.0, 1.0])
        np.testing.assert_allclose(rrc.MultisetMedian(4).process_batch(x),
                                   median_ref(x, 4), rtol=1e-12)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-2.0, 6.0, 1.0, -8.0, 0.0, 8.0, -1.0])
        np.testing.assert_allclose(rrc.MultisetMedian(k).process_batch(x),
                                   median_ref(x, k), rtol=1e-12)

    def test_large_array_against_reference(self):
        np.random.seed(42)
        x = np.random.randn(1000)
        k = 15
        np.testing.assert_allclose(rrc.MultisetMedian(k).process_batch(x),
                                   median_ref(x, k), rtol=1e-10)

    def test_element_entering_equals_leaving(self):
        x = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(rrc.MultisetMedian(3).process_batch(x),
                                   median_ref(x, 3), rtol=1e-12)

    def test_window_larger_than_array(self):
        x = np.array([3.0, 1.0, 4.0, 1.0])
        np.testing.assert_allclose(rrc.MultisetMedian(20).process_batch(x),
                                   median_ref(x, 20), rtol=1e-12)

    def test_nan_does_not_contribute(self):
        # window=3, [1, 2, NaN, 4]: at NaN window=[1,2,NaN] -> median=1.5
        x = np.array([1.0, 2.0, np.nan, 4.0])
        out = rrc.MultisetMedian(3).process_batch(x)
        assert np.isclose(out[2], 1.5, atol=1e-12)
        assert np.isclose(out[3], 3.0, atol=1e-12)  # window=[2,NaN,4] -> median([2,4])=3

    def test_empty_input(self):
        out = rrc.MultisetMedian(3).process_batch(np.array([]))
        assert len(out) == 0 and out.dtype == np.float64

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.MultisetMedian(0)

    def test_rejects_none_window(self):
        with pytest.raises(TypeError):
            rrc.MultisetMedian(None)

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.MultisetMedian(2).process_batch(np.ones((2, 3)))

    def test_integer_input_converted_to_float64(self):
        out = rrc.MultisetMedian(2).process_batch(np.array([1, 2, 3], dtype=np.int32))
        assert out.dtype == np.float64


# ── SlidingMean ────────────────────────────────────────────────────────────────

class TestSlidingMean:

    def test_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x)
        nan_allclose(out, [1.0, 1.5, 2.0, 3.0, 4.0])

    def test_window_size_1_identity(self):
        x = np.array([3.0, 7.0, -1.0])
        nan_allclose(rrc.SlidingMean(1).process_batch(x), x)

    def test_window_larger_than_array(self):
        nan_allclose(rrc.SlidingMean(10).process_batch(np.array([2.0, 4.0, 6.0])),
                     [2.0, 3.0, 4.0])

    def test_constant_sequence(self):
        out = rrc.SlidingMean(3).process_batch(np.full(6, 5.0))
        np.testing.assert_allclose(out, 5.0, atol=1e-12)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_against_naive_reference(self, k):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 10.0, 7.0, 7.0, 8.0])
        out = rrc.SlidingMean(k).process_batch(x)
        np.testing.assert_allclose(out, mean_ref(x, k), rtol=1e-12, atol=1e-12,
                                   equal_nan=True)

    def test_nan_does_not_contribute(self):
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x)
        assert np.isclose(out[2], 2.0, atol=1e-12)  # mean([1,3]) = 2.0

    def test_nan_advances_window(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x)
        assert np.isclose(out[3], 3.0, atol=1e-12)  # mean([2,4]) = 3.0

    def test_all_nan_returns_nan(self):
        out = rrc.SlidingMean(2).process_batch(np.array([np.nan, np.nan, np.nan]))
        assert np.all(np.isnan(out))

    def test_empty_input(self):
        out = rrc.SlidingMean(3).process_batch(np.array([]))
        assert len(out) == 0 and out.dtype == np.float64

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="Window length must be greater than 0"):
            rrc.SlidingMean(0)

    def test_rejects_2d_input(self):
        with pytest.raises(RuntimeError, match="Input must be 1D array"):
            rrc.SlidingMean(2).process_batch(np.ones((2, 3)))

    def test_integer_input_converted_to_float64(self):
        out = rrc.SlidingMean(2).process_batch(np.array([1, 2, 3, 4], dtype=np.int32))
        assert out.dtype == np.float64

    def test_min_periods_masks_warmup(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x, min_periods=3)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert np.isclose(out[2], 2.0, atol=1e-12)

    def test_min_periods_with_nan_input(self):
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        out = rrc.SlidingMean(3).process_batch(x, min_periods=2)
        assert np.isnan(out[1])               # 1 non-NaN < 2
        assert np.isclose(out[2], 2.0, atol=1e-12)  # 2 non-NaN >= 2


# ── SlidingMoments ─────────────────────────────────────────────────────────────

class TestSlidingMoments:

    def test_initial_state_is_nan(self):
        sm = rrc.SlidingMoments(4)
        assert sm.current_size() == 0
        assert math.isnan(sm.get_skewness())
        assert math.isnan(sm.get_kurtosis())

    def test_symmetric_window_skewness_is_zero(self):
        sm = rrc.SlidingMoments(3)
        for v in [1.0, 2.0, 3.0]:
            sm.update(v)
        assert pytest.approx(sm.get_skewness(), abs=1e-12) == 0.0
        assert math.isnan(sm.get_kurtosis())  # n=3 < 4

    def test_known_kurtosis_uniform_window(self):
        # [1, 2, 3, 4]: excess kurtosis = -1.2
        sm = rrc.SlidingMoments(4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            sm.update(v)
        assert pytest.approx(sm.get_kurtosis(), abs=1e-10) == -1.2

    def test_small_window_skewness_always_nan(self):
        # window=2 -> at most 2 values -> skewness (needs >=3) always NaN
        sm = rrc.SlidingMoments(2)
        for v in [1.0, 2.0, 3.0]:
            sm.update(v)
            assert math.isnan(sm.get_skewness())
            assert math.isnan(sm.get_kurtosis())

    def test_window_size_1_both_moments_always_nan(self):
        sm = rrc.SlidingMoments(1)
        sm.update(5.0)
        assert sm.current_size() == 1
        assert math.isnan(sm.get_skewness())
        assert math.isnan(sm.get_kurtosis())

    def test_nan_advances_window_and_reduces_size(self):
        sm = rrc.SlidingMoments(4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            sm.update(v)
        assert sm.current_size() == 4
        assert not math.isnan(sm.get_kurtosis())

        sm.update(float('nan'))
        assert sm.current_size() == 3
        assert not math.isnan(sm.get_skewness())
        assert math.isnan(sm.get_kurtosis())  # n=3 < 4

    def test_nan_does_not_corrupt_state(self):
        sm = rrc.SlidingMoments(3)
        for v in [10.0, 20.0, 30.0]:
            sm.update(v)
        skew_before = sm.get_skewness()

        for _ in range(3):
            sm.update(float('nan'))
        assert sm.current_size() == 0

        for v in [10.0, 20.0, 30.0]:
            sm.update(v)
        assert pytest.approx(sm.get_skewness(), abs=1e-10) == skew_before

    def test_randomized_fuzzing_against_pandas(self):
        np.random.seed(42)
        window_size = int(np.random.randint(4, 20))
        data = np.random.randn(1000) * 50.0
        data[np.random.rand(1000) < 0.15] = np.nan
        s = pd.Series(data)
        expected_skew = s.rolling(window=window_size, min_periods=3).skew()
        expected_kurt = s.rolling(window=window_size, min_periods=4).kurt()
        expected_count = s.rolling(window=window_size, min_periods=0).count()

        sm = rrc.SlidingMoments(window_size)
        for i, val in enumerate(data):
            sm.update(val)
            assert sm.current_size() == expected_count[i], f"count mismatch at {i}"
            cpp_skew, pd_skew = sm.get_skewness(), expected_skew[i]
            if pd.isna(pd_skew):
                assert math.isnan(cpp_skew), f"skewness should be NaN at {i}"
            else:
                assert pytest.approx(cpp_skew, rel=1e-4, abs=1e-6) == pd_skew, \
                    f"skewness mismatch at {i}"
            cpp_kurt, pd_kurt = sm.get_kurtosis(), expected_kurt[i]
            if pd.isna(pd_kurt):
                assert math.isnan(cpp_kurt), f"kurtosis should be NaN at {i}"
            else:
                assert pytest.approx(cpp_kurt, rel=1e-4, abs=1e-6) == pd_kurt, \
                    f"kurtosis mismatch at {i}"


# ── SlidingCovariance ──────────────────────────────────────────────────────────

class TestSlidingCovariance:

    def test_initial_state_is_nan(self):
        sc = rrc.SlidingCovariance(3)
        assert math.isnan(sc.get_covariance())
        assert math.isnan(sc.get_correlation())
        assert math.isnan(sc.get_mean_x())
        assert math.isnan(sc.get_mean_y())

    def test_perfect_positive_correlation(self):
        sc = rrc.SlidingCovariance(3)
        for xi, yi in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]:
            sc.update(xi, yi)
        assert pytest.approx(sc.get_correlation(), abs=1e-12) == 1.0
        assert pytest.approx(sc.get_covariance(), abs=1e-12) == 2.0

    def test_perfect_negative_correlation(self):
        sc = rrc.SlidingCovariance(3)
        for xi, yi in [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]:
            sc.update(xi, yi)
        assert pytest.approx(sc.get_correlation(), abs=1e-12) == -1.0

    def test_constant_x_correlation_is_nan(self):
        # std_dev(x) = 0 -> correlation = NaN, covariance = 0
        sc = rrc.SlidingCovariance(4)
        for yi in [1.0, 2.0, 3.0, 4.0]:
            sc.update(5.0, yi)
        assert math.isnan(sc.get_correlation())
        assert pytest.approx(sc.get_covariance(), abs=1e-12) == 0.0
        assert pytest.approx(sc.get_mean_x(), abs=1e-12) == 5.0

    def test_window_expiry_removes_old_pairs(self):
        sc = rrc.SlidingCovariance(2)
        sc.update(1.0, 1.0)
        sc.update(2.0, 2.0)
        sc.update(10.0, 10.0)  # window: [(2,2),(10,10)]
        assert pytest.approx(sc.get_mean_x(), abs=1e-12) == 6.0
        assert pytest.approx(sc.get_correlation(), abs=1e-12) == 1.0

    def test_nan_pair_skipped_not_added(self):
        sc = rrc.SlidingCovariance(3)
        for xi, yi in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]:
            sc.update(xi, yi)
        sc.update(math.nan, 5.0)
        assert pytest.approx(sc.get_covariance(), abs=1e-12) == 1.0
        assert pytest.approx(sc.get_correlation(), abs=1e-12) == 1.0

    def test_single_pair_covariance_is_nan(self):
        sc = rrc.SlidingCovariance(3)
        sc.update(1.0, 2.0)
        assert math.isnan(sc.get_covariance())

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_covariance_against_numpy_reference(self, k):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 10.0, 7.0, 7.0, 8.0])
        y = np.array([1.0, 3.0, -1.0, 4.0, 2.0, 6.0, 5.0, 0.0])
        out = rrc.SlidingCovariance(k).process_covariance_batch(x, y)
        nan_allclose(out, cov_ref(x, y, k), rtol=1e-11, atol=1e-11)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_correlation_against_numpy_reference(self, k):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 10.0, 7.0, 7.0, 8.0])
        y = np.array([1.0, 3.0, -1.0, 4.0, 2.0, 6.0, 5.0, 0.0])
        out = rrc.SlidingCovariance(k).process_correlation_batch(x, y)
        nan_allclose(out, cor_ref(x, y, k), rtol=1e-11, atol=1e-11)

    def test_random_nan_fuzzing_against_reference(self):
        np.random.seed(123)
        x = np.random.randn(500)
        y = np.random.randn(500)
        x[np.random.rand(500) < 0.15] = np.nan
        y[np.random.rand(500) < 0.15] = np.nan
        k = 10
        cov_out = rrc.SlidingCovariance(k).process_covariance_batch(x, y)
        cor_out = rrc.SlidingCovariance(k).process_correlation_batch(x, y)
        nan_allclose(cov_out, cov_ref(x, y, k), rtol=1e-9, atol=1e-9)
        nan_allclose(cor_out, cor_ref(x, y, k), rtol=1e-9, atol=1e-9)

    def test_process_batch_rejects_length_mismatch(self):
        sc = rrc.SlidingCovariance(3)
        with pytest.raises(RuntimeError, match="same length"):
            sc.process_covariance_batch(np.array([1.0, 2.0]), np.array([1.0]))

    def test_process_batch_rejects_2d_input(self):
        sc = rrc.SlidingCovariance(3)
        with pytest.raises(RuntimeError):
            sc.process_covariance_batch(np.ones((2, 3)), np.ones(3))

    def test_empty_input(self):
        out = rrc.SlidingCovariance(3).process_covariance_batch(np.array([]), np.array([]))
        assert len(out) == 0 and out.dtype == np.float64
