#include "SlidingCovariance.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(SlidingCovarianceTest, InitialStateIsNaN) {
  SlidingCovariance sc(3);
  EXPECT_TRUE(std::isnan(sc.get_covariance()));
  EXPECT_TRUE(std::isnan(sc.get_correlation()));
  EXPECT_TRUE(std::isnan(sc.get_mean_x()));
  EXPECT_TRUE(std::isnan(sc.get_mean_y()));
}

TEST(SlidingCovarianceTest, PerfectPositiveCorrelation) {
  // y = 2x correlation = 1.0, cov = 2 * var(x)
  SlidingCovariance sc(3);
  sc.update(1.0, 2.0);
  sc.update(2.0, 4.0);
  sc.update(3.0, 6.0);
  EXPECT_NEAR(sc.get_correlation(), 1.0, 1e-12);
  EXPECT_NEAR(sc.get_covariance(), 2.0, 1e-12); // cov([1,2,3],[2,4,6]) = 2
}

TEST(SlidingCovarianceTest, PerfectNegativeCorrelation) {
  // y = -x + 4 correlation = -1.0
  SlidingCovariance sc(3);
  sc.update(1.0, 3.0);
  sc.update(2.0, 2.0);
  sc.update(3.0, 1.0);
  EXPECT_NEAR(sc.get_correlation(), -1.0, 1e-12);
}

TEST(SlidingCovarianceTest, SlidingWindowExpiry) {
  // After window slides, old values should not contribute
  SlidingCovariance sc(2);
  sc.update(1.0, 1.0);
  sc.update(2.0, 2.0);
  sc.update(10.0, 10.0); // window now: [(2,2),(10,10)]
  EXPECT_NEAR(sc.get_mean_x(), 6.0, 1e-12);
  EXPECT_NEAR(sc.get_correlation(), 1.0, 1e-12);
}

TEST(SlidingCovarianceTest, NanPairSkipped) {
  SlidingCovariance sc(3);
  sc.update(1.0, 2.0);
  sc.update(2.0, 4.0);
  sc.update(3.0, 6.0);

  // NaN pair: removes oldest (1,2), NaN not added  valid: [(2,4),(3,6)]
  // cov([2,3],[4,6]) = 1.0, corr = 1.0
  sc.update(std::numeric_limits<double>::quiet_NaN(), 5.0);
  EXPECT_NEAR(sc.get_covariance(), 1.0, 1e-12);
  EXPECT_NEAR(sc.get_correlation(), 1.0, 1e-12);
}

TEST(SlidingCovarianceTest, SinglePairCovarianceIsNaN) {
  SlidingCovariance sc(3);
  sc.update(1.0, 2.0);
  EXPECT_TRUE(std::isnan(sc.get_covariance()));
  EXPECT_TRUE(std::isnan(sc.get_correlation()));
  EXPECT_EQ(sc.current_size(), 1);
}

TEST(SlidingCovarianceTest, ConstantXCorrelationIsNaN) {
  // x is constant -> std_dev(x) = 0 -> correlation = cov / (0 * sd_y) = NaN
  // covariance itself is 0 (x never deviates from its mean)
  SlidingCovariance sc(4);
  sc.update(5.0, 1.0);
  sc.update(5.0, 2.0);
  sc.update(5.0, 3.0);
  sc.update(5.0, 4.0);
  EXPECT_TRUE(std::isnan(sc.get_correlation()));
  EXPECT_NEAR(sc.get_covariance(), 0.0, 1e-12);
  EXPECT_NEAR(sc.get_mean_x(), 5.0, 1e-12);
}