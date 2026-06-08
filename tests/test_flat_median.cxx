#include "FlatMedian.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

TEST(FlatMedianTest, ThrowsOnZeroWindowSize) {
  EXPECT_THROW(FlatMedian(0), std::invalid_argument);
}

TEST(FlatMedianTest, InitialStateIsNaN) {
  FlatMedian m(3);
  EXPECT_EQ(m.current_size(), 0U);
  EXPECT_TRUE(std::isnan(m.get_median()));
}

TEST(FlatMedianTest, WarmUpAndSliding) {
  FlatMedian m(3);

  m.update(10.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 10.0);

  m.update(20.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 15.0);  // [10, 20]

  m.update(5.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 10.0);  // [5, 10, 20]

  m.update(4.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 5.0);   // [4, 5, 20] after evicting 10
}

TEST(FlatMedianTest, EvenWindowMedian) {
  FlatMedian m(4);
  m.update(4.0);
  m.update(3.0);
  m.update(2.0);
  m.update(1.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 2.5);  // [1, 2, 3, 4]
}

TEST(FlatMedianTest, WindowSize1) {
  FlatMedian m(1);
  m.update(5.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 5.0);
  m.update(3.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 3.0);
}

TEST(FlatMedianTest, HandlesDuplicates) {
  FlatMedian m(3);
  m.update(5.0);
  m.update(5.0);
  m.update(5.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 5.0);

  m.update(1.0);                           // evict 5 -> [1, 5, 5]
  EXPECT_DOUBLE_EQ(m.get_median(), 5.0);

  m.update(1.0);                           // evict 5 -> [1, 1, 5]
  EXPECT_DOUBLE_EQ(m.get_median(), 1.0);
}

TEST(FlatMedianTest, SkipDoesNotContributeToWindow) {
  // window=3, sequence [1, 2, skip, 4]:
  // after skip:      window=[1,2,NaN] -> valid=[1,2], median=1.5
  // after update(4): window=[2,NaN,4] -> valid=[2,4], median=3.0
  FlatMedian m(3);
  m.update(1.0);
  m.update(2.0);

  m.skip();
  EXPECT_DOUBLE_EQ(m.get_median(), 1.5);
  EXPECT_EQ(m.current_size(), 2U);

  m.update(4.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 3.0);
  EXPECT_EQ(m.current_size(), 2U);
}

TEST(FlatMedianTest, CrtpInterfaceMatchesMedian) {
  FlatMedian m(3);
  RollingMetric<FlatMedian> &base = m;

  EXPECT_TRUE(std::isnan(base.get_value()));

  base.update(7.0);
  base.update(1.0);
  base.update(4.0);  // [1, 4, 7]

  EXPECT_EQ(base.current_size(), 3U);
  EXPECT_DOUBLE_EQ(base.get_value(), m.get_median());
}

TEST(FlatMedianTest, ProcessBatch) {
  FlatMedian m(3);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double input[] = {1.0, 3.0, 2.0, nan, 5.0};
  double output[5];
  m.process_batch(input, 5, output, /*min_periods=*/2);

  EXPECT_TRUE(std::isnan(output[0]));     // size=1 < 2
  EXPECT_DOUBLE_EQ(output[1], 2.0);      // [1,3] -> 2
  EXPECT_DOUBLE_EQ(output[2], 2.0);      // [1,2,3] -> 2
  EXPECT_DOUBLE_EQ(output[3], 2.5);      // [2,3,NaN] -> valid=[2,3] -> 2.5
  EXPECT_DOUBLE_EQ(output[4], 3.5);      // [2,NaN,5] -> valid=[2,5] -> 3.5
}
