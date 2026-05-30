#include "FlatMedian.hpp"
#include "TwoHeapMedian.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

TEST(TwoHeapMedianTest, ThrowsOnZeroWindowSize) {
  EXPECT_THROW(TwoHeapMedian(0), std::invalid_argument);
}

TEST(TwoHeapMedianTest, InitialStateIsNaN) {
  TwoHeapMedian m(3);
  EXPECT_EQ(m.current_size(), 0U);
  EXPECT_TRUE(std::isnan(m.get_median()));
}

TEST(TwoHeapMedianTest, WarmUpAndSliding) {
  TwoHeapMedian m(3);
  m.update(10.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 10.0);

  m.update(20.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 15.0);  // [10, 20]

  m.update(5.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 10.0);  // [5, 10, 20]

  m.update(4.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 5.0);   // [4, 5, 20]
}

TEST(TwoHeapMedianTest, EvenWindowAndWindowSize2) {
  TwoHeapMedian m(2);
  m.update(1.0);
  m.update(2.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 1.5);
  m.update(3.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 2.5);
}

TEST(TwoHeapMedianTest, WindowSize1) {
  TwoHeapMedian m(1);
  m.update(5.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 5.0);
  m.update(3.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 3.0);
  m.update(8.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 8.0);
}

TEST(TwoHeapMedianTest, HandlesDuplicatesWithLazyDeletion) {
  // Multiple identical values exercise the pending-map eviction path
  TwoHeapMedian m(3);
  m.update(5.0);
  m.update(5.0);
  m.update(5.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 5.0);

  m.update(3.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 5.0);   // [3, 5, 5]

  m.update(1.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 3.0);   // [1, 3, 5]

  m.update(2.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 2.0);   // [1, 2, 3]
}

TEST(TwoHeapMedianTest, SkipDoesNotContributeToWindow) {
  // Same semantics as MultisetMedian / FlatMedian
  TwoHeapMedian m(3);
  m.update(1.0);
  m.update(2.0);

  m.skip();
  EXPECT_DOUBLE_EQ(m.get_median(), 1.5);
  EXPECT_EQ(m.current_size(), 2U);

  m.update(4.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 3.0);   // [2, NaN, 4] -> valid=[2,4]
  EXPECT_EQ(m.current_size(), 2U);
}

TEST(TwoHeapMedianTest, MultipleSkips) {
  TwoHeapMedian m(4);
  m.update(1.0);
  m.skip();
  m.skip();
  m.update(2.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 1.5);
  EXPECT_EQ(m.current_size(), 2U);
}

TEST(TwoHeapMedianTest, CrtpInterfaceMatchesMedian) {
  TwoHeapMedian m(3);
  RollingMetric<TwoHeapMedian> &base = m;

  EXPECT_TRUE(std::isnan(base.get_value()));

  base.update(7.0);
  base.update(1.0);
  base.update(4.0);  // [1, 4, 7]

  EXPECT_EQ(base.current_size(), 3U);
  EXPECT_DOUBLE_EQ(base.get_value(), m.get_median());
}

TEST(TwoHeapMedianTest, AgreeWithFlatMedian) {
  // Cross-check: every step must match the reference O(n) implementation
  const std::vector<double> seq = {5, 3, 1, 7, 2, 9, 4, 6, 8, 0,
                                   5, 5, 5, 2, 9, 1, 3, 7, 4};
  FlatMedian ref(5);
  TwoHeapMedian m(5);

  for (double v : seq) {
    ref.update(v);
    m.update(v);
    EXPECT_DOUBLE_EQ(m.get_median(), ref.get_median());
  }
}
