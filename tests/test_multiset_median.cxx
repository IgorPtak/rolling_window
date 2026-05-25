#include "MultisetMedian.hpp"

#include <cmath>
#include <gtest/gtest.h>

TEST(MultisetMedianTest, HandlesInitialization) {
  MultisetMedian sm(3);
  EXPECT_EQ(sm.current_size(), 0);

  EXPECT_TRUE(std::isnan(sm.get_median()));
}

TEST(MultisetMedianTest, HandlesWarmUpPhase) {
  MultisetMedian sm(3);

  sm.update(10.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 10.0);

  sm.update(20.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 15.0);

  sm.update(5.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 10.0); //  5, 10, 20
}

TEST(MultisetMedianTest, HandlesWindowShifting) {
  MultisetMedian sm(3);
  sm.update(1.0);
  sm.update(2.0);
  sm.update(3.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 2.0);

  sm.update(4.0); // 2, 3, 4
  EXPECT_DOUBLE_EQ(sm.get_median(), 3.0);

  sm.update(0.0); // 3, 4, 0 -> 0, 3, 4
  EXPECT_DOUBLE_EQ(sm.get_median(), 3.0);
}

TEST(MultisetMedianTest, HandlesDuplicateValues) {
  MultisetMedian sm(4);
  sm.update(5.0);
  sm.update(5.0);
  sm.update(5.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 5.0);

  sm.update(5.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 5.0);
}

TEST(MultisetMedianTest, CrtpInterfaceMatchesMedian) {
  MultisetMedian sm(3);
  RollingMetric<MultisetMedian> &base = sm;

  EXPECT_TRUE(std::isnan(base.get_value()));

  base.update(7.0);
  base.update(1.0);
  base.update(4.0);

  EXPECT_EQ(base.current_size(), 3U);
  EXPECT_DOUBLE_EQ(base.get_value(), sm.get_median());
}

TEST(MultisetMedianTest, HandlesEvenWindowDescendingFill) {
  MultisetMedian sm(4);
  sm.update(4.0);
  sm.update(3.0);
  sm.update(2.0);
  sm.update(1.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 2.5);
}

TEST(MultisetMedianTest, HandlesWindowSize2Sliding) {
  MultisetMedian sm(2);
  sm.update(1.0);
  sm.update(2.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 1.5);
  sm.update(3.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 2.5);
}

TEST(MultisetMedianTest, WindowSize1Identity) {
  MultisetMedian sm(1);
  sm.update(5.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 5.0);
  sm.update(3.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 3.0);
  sm.update(8.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 8.0);
}

TEST(MultisetMedianTest, NanDoesNotContributeToWindow) {
  // window=3, sequence [1, 2, skip, 4]:
  // after skip:     window=[1,2,NaN] -> valid=[1,2], median=1.5
  // after update(4): window=[2,NaN,4] -> valid=[2,4], median=3.0
  MultisetMedian sm(3);
  sm.update(1.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 1.0);
  EXPECT_EQ(sm.current_size(), 1U);

  sm.update(2.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 1.5);
  EXPECT_EQ(sm.current_size(), 2U);

  sm.skip();
  EXPECT_DOUBLE_EQ(sm.get_median(), 1.5);
  EXPECT_EQ(sm.current_size(), 2U);

  sm.update(4.0);
  EXPECT_DOUBLE_EQ(sm.get_median(), 3.0);
  EXPECT_EQ(sm.current_size(), 2U);
}