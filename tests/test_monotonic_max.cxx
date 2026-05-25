#include "MonotonicMax.hpp"

#include <cmath>
#include <gtest/gtest.h>

TEST(MonotonicMaxTest, BasicFunctionality) {
  MonotonicMax mm(3);

  mm.update(10.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 10.0);

  mm.update(20.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 20.0);

  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 20.0); // [10, 20, 5]
}

TEST(MonotonicMaxTest, MaxExitsWindow) {
  MonotonicMax mm(3);
  mm.update(50.0);
  mm.update(10.0);
  mm.update(20.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 50.0);

  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 20.0);
}

TEST(MonotonicMaxTest, StrictlyDecreasing) {
  MonotonicMax mm(3);
  mm.update(100.0);
  mm.update(90.0);
  mm.update(80.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 100.0);

  mm.update(70.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 90.0);
}

TEST(MonotonicMaxTest, Duplicates) {
  MonotonicMax mm(2);
  mm.update(10.0);
  mm.update(10.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 10.0);

  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 10.0);
}

TEST(MonotonicMaxTest, CrtpInterfaceMatchesMax) {
  MonotonicMax mm(3);
  RollingMetric<MonotonicMax> &base = mm;

  EXPECT_TRUE(std::isnan(base.get_value()));

  base.update(1.0);
  base.update(9.0);
  base.update(2.0);

  EXPECT_EQ(base.current_size(), 3U);
  EXPECT_DOUBLE_EQ(base.get_value(), mm.get_max());
}

TEST(MonotonicMaxTest, InitialStateIsNaN) {
  MonotonicMax mm(3);
  EXPECT_TRUE(std::isnan(mm.get_max()));
  EXPECT_EQ(mm.current_size(), 0);
}

TEST(MonotonicMaxTest, WindowSize1Identity) {
  MonotonicMax mm(1);
  mm.update(5.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 5.0);
  mm.update(3.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 3.0);
  mm.update(8.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 8.0);
  mm.update(1.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 1.0);
}

TEST(MonotonicMaxTest, NanDoesNotContributeToWindow) {
  MonotonicMax mm(2);
  mm.update(1.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 1.0);
  mm.update(2.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 2.0);
  mm.skip();
  EXPECT_DOUBLE_EQ(mm.get_max(), 2.0);
  mm.update(1.0);
  EXPECT_DOUBLE_EQ(mm.get_max(), 1.0);
}

TEST(MonotonicMaxTest, NanAtStartReturnsNan) {
  MonotonicMax mm(3);
  mm.skip();
  EXPECT_TRUE(std::isnan(mm.get_max()));
  EXPECT_EQ(mm.current_size(), 0);
}