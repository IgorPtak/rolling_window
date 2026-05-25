#include "SlidingWelfordRing.hpp"

#include <cmath>
#include <gtest/gtest.h>

TEST(SlidingWelfordRingTest, WarmUpPhase) {
  SlidingWelfordRing sw(3);

  sw.update(10.0);
  EXPECT_DOUBLE_EQ(sw.get_mean(), 10.0);
  EXPECT_TRUE(std::isnan(sw.get_variance()));

  sw.update(20.0);
  // Mean: 15, Variance: ((10-15)^2 + (20-15)^2) / 1 = 50
  EXPECT_DOUBLE_EQ(sw.get_mean(), 15.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 50.0);
}

TEST(SlidingWelfordRingTest, ConstantValues) {
  SlidingWelfordRing sw(10);
  for (int i = 0; i < 20; ++i) {
    sw.update(5.0);
  }
  EXPECT_DOUBLE_EQ(sw.get_mean(), 5.0);
  EXPECT_NEAR(sw.get_std_dev(), 0.0, 1e-9);
}

TEST(SlidingWelfordRingTest, BufferWrapAround) {
  SlidingWelfordRing sw(2);

  sw.update(10.0);
  sw.update(20.0); // [10, 20]

  sw.update(30.0); // [20, 30]
  EXPECT_DOUBLE_EQ(sw.get_mean(), 25.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 50.0);

  sw.update(40.0); // [30, 40]
  EXPECT_DOUBLE_EQ(sw.get_mean(), 35.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 50.0);
}

TEST(SlidingWelfordRingTest, PrecisionTest) {
  SlidingWelfordRing sw(3);
  sw.update(1000000.1);
  sw.update(1000000.2);
  sw.update(1000000.3);

  EXPECT_NEAR(sw.get_mean(), 1000000.2, 1e-7);
  EXPECT_NEAR(sw.get_variance(), 0.01, 1e-9);
}

TEST(SlidingWelfordRingTest, CrtpInterfaceMatchesVariance) {
  SlidingWelfordRing sw(2);
  RollingMetric<SlidingWelfordRing> &base = sw;

  base.update(3.0);
  EXPECT_TRUE(std::isnan(base.get_value()));

  base.update(5.0);
  EXPECT_EQ(base.current_size(), 2U);
  EXPECT_DOUBLE_EQ(base.get_value(), sw.get_variance());
}

TEST(SlidingWelfordRingTest, NanDoesNotContributeToWindow) {
  // window=3, sequence [10, 20, skip, 30]:
  // after skip:    valid=[10,20], mean=15, var=50, size=2
  // after update(30): 10 expires -> valid=[20,30], mean=25, var=50, size=2
  SlidingWelfordRing sw(3);

  sw.update(10.0);
  EXPECT_DOUBLE_EQ(sw.get_mean(), 10.0);
  EXPECT_EQ(sw.current_size(), 1U);

  sw.update(20.0);
  EXPECT_DOUBLE_EQ(sw.get_mean(), 15.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 50.0);
  EXPECT_EQ(sw.current_size(), 2U);

  sw.skip();
  EXPECT_DOUBLE_EQ(sw.get_mean(), 15.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 50.0);
  EXPECT_EQ(sw.current_size(), 2U);

  sw.update(30.0);
  EXPECT_DOUBLE_EQ(sw.get_mean(), 25.0);
  EXPECT_DOUBLE_EQ(sw.get_variance(), 50.0);
  EXPECT_EQ(sw.current_size(), 2U);
}
