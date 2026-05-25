#include "MonotonicMin.hpp"

#include <cmath>
#include <gtest/gtest.h>

TEST(MonotonicMinTest, BasicFunctionality) {
  MonotonicMin mm(3);

  mm.update(10.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 10.0);

  mm.update(20.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 10.0);

  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 5.0); // [10, 20, 5]
}

TEST(MonotonicMinTest, MinExitsWindow) {
  MonotonicMin mm(3);
  mm.update(50.0);
  mm.update(10.0);
  mm.update(20.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 10.0);

  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 5.0);
}

TEST(MonotonicMinTest, StrictlyDecreasing) {
  MonotonicMin mm(3);
  mm.update(100.0);
  mm.update(90.0);
  mm.update(80.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 80.0);

  mm.update(70.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 70.0);
}

TEST(MonotonicMinTest, Duplicates) {
  MonotonicMin mm(2);
  mm.update(10.0);
  mm.update(10.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 10.0);

  mm.update(20.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 10.0);
}

TEST(MonotonicMinTest, CrtpInterfaceMatchesMin) {
  MonotonicMin mm(3);
  RollingMetric<MonotonicMin> &base = mm;

  EXPECT_TRUE(std::isnan(base.get_value()));

  base.update(1.0);
  base.update(9.0);
  base.update(2.0);

  EXPECT_EQ(base.current_size(), 3U);
  EXPECT_DOUBLE_EQ(base.get_value(), mm.get_min());
}

TEST(MonotonicMinTest, InitialStateIsNaN) {
  MonotonicMin mm(3);
  EXPECT_TRUE(std::isnan(mm.get_min()));
  EXPECT_EQ(mm.current_size(), 0U);
}

TEST(MonotonicMinTest, WindowSize1Identity) {
  MonotonicMin mm(1);
  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 5.0);
  mm.update(3.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 3.0);
  mm.update(8.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 8.0);
  mm.update(1.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 1.0);
}

TEST(MonotonicMinTest, NanDoesNotContributeToWindow) {
  MonotonicMin mm(2);
  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 5.0);
  mm.update(1.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 1.0);
  mm.skip();
  EXPECT_DOUBLE_EQ(mm.get_min(), 1.0);
  EXPECT_EQ(mm.current_size(), 1U);
  mm.update(9.0);
  EXPECT_DOUBLE_EQ(mm.get_min(), 9.0);
  EXPECT_EQ(mm.current_size(), 1U);
}

TEST(MonotonicMinTest, NanAtStartReturnsNan) {
  MonotonicMin mm(3);
  mm.skip();
  EXPECT_TRUE(std::isnan(mm.get_min()));
  EXPECT_EQ(mm.current_size(), 0U);
}
