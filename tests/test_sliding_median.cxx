#include "FlatMedian.hpp"
#include "SlidingMedian.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

TEST(SlidingMedianTest, ThrowsOnZeroWindowSize) {
  EXPECT_THROW(SlidingMedian(0), std::invalid_argument);
}

TEST(SlidingMedianTest, InitialStateIsNaN) {
  EXPECT_TRUE(std::isnan(SlidingMedian(10).get_median()));
  EXPECT_TRUE(std::isnan(SlidingMedian(1000).get_median()));
  EXPECT_TRUE(std::isnan(SlidingMedian(3000).get_median()));
}

TEST(SlidingMedianTest, DispatchBoundariesCleanData) {
  // kFlatClean=600: <=600 -> Flat, 601..2000 -> Multiset, >2000 -> TwoHeap
  for (std::size_t w : {600u, 601u, 2000u, 2001u}) {
    SlidingMedian m(w);
    m.update(1.0);
    m.update(3.0);
    m.update(2.0);
    EXPECT_DOUBLE_EQ(m.get_median(), 2.0) << "window=" << w;
  }
}

TEST(SlidingMedianTest, DispatchBoundariesNaNData) {
  // kFlatNaN=1500: <=1500 -> Flat, >1500 -> TwoHeap
  for (std::size_t w : {1500u, 1501u}) {
    SlidingMedian m(w, /*expect_nan=*/true);
    m.update(1.0);
    m.update(3.0);
    m.update(2.0);
    EXPECT_DOUBLE_EQ(m.get_median(), 2.0) << "window=" << w;
  }
}

TEST(SlidingMedianTest, WarmUpAndSliding) {
  SlidingMedian m(3);
  m.update(1.0);
  m.update(2.0);
  m.update(3.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 2.0);

  m.update(4.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 3.0);  // evict 1 -> [2, 3, 4]
}

TEST(SlidingMedianTest, SkipDoesNotContributeToWindow) {
  SlidingMedian m(3);
  m.update(1.0);
  m.update(2.0);

  m.skip();
  EXPECT_DOUBLE_EQ(m.get_median(), 1.5);
  EXPECT_EQ(m.current_size(), 2U);

  m.update(4.0);
  EXPECT_DOUBLE_EQ(m.get_median(), 3.0);  // [2, NaN, 4] -> valid=[2,4]
}

TEST(SlidingMedianTest, CrtpInterfaceMatchesMedian) {
  SlidingMedian m(3);
  RollingMetric<SlidingMedian> &base = m;

  base.update(7.0);
  base.update(1.0);
  base.update(4.0);  // [1, 4, 7]

  EXPECT_EQ(base.current_size(), 3U);
  EXPECT_DOUBLE_EQ(base.get_value(), m.get_median());
}

TEST(SlidingMedianTest, ProcessBatch) {
  SlidingMedian m(3);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double input[] = {1.0, 3.0, 2.0, nan, 5.0};
  double output[5];
  m.process_batch(input, 5, output, /*min_periods=*/2);

  EXPECT_TRUE(std::isnan(output[0]));     // size=1 < 2
  EXPECT_DOUBLE_EQ(output[1], 2.0);      // [1,3]
  EXPECT_DOUBLE_EQ(output[2], 2.0);      // [1,2,3]
  EXPECT_DOUBLE_EQ(output[3], 2.5);      // [2,3,NaN] -> valid=[2,3]
  EXPECT_DOUBLE_EQ(output[4], 3.5);      // [2,NaN,5] -> valid=[2,5]
}

// Cross-validation: all dispatch paths must agree with the canonical implementation

static const std::vector<double> kSeq = {5, 3, 1, 7, 2, 9, 4, 6, 8, 0,
                                         5, 5, 5, 2, 9, 1, 3, 7, 4};

TEST(SlidingMedianTest, MatchesFlatMedian_FlatPath) {
  FlatMedian ref(7);
  SlidingMedian m(7);  // <= kFlatClean(600) -> Flat
  for (double v : kSeq) {
    ref.update(v); m.update(v);
    EXPECT_DOUBLE_EQ(m.get_median(), ref.get_median());
  }
}

TEST(SlidingMedianTest, MatchesFlatMedian_MultisetPath) {
  FlatMedian ref(700);
  SlidingMedian m(700);  // kFlatClean < 700 <= kHeapClean -> Multiset
  for (double v : kSeq) {
    ref.update(v); m.update(v);
    EXPECT_DOUBLE_EQ(m.get_median(), ref.get_median());
  }
}

TEST(SlidingMedianTest, MatchesFlatMedian_TwoHeapPath) {
  FlatMedian ref(2500);
  SlidingMedian m(2500);  // > kHeapClean(2000) -> TwoHeap
  for (double v : kSeq) {
    ref.update(v); m.update(v);
    EXPECT_DOUBLE_EQ(m.get_median(), ref.get_median());
  }
}

TEST(SlidingMedianTest, MatchesFlatMedian_ExpectNaNFlatPath) {
  FlatMedian ref(100);
  SlidingMedian m(100, /*expect_nan=*/true);  // <= kFlatNaN(1500) -> Flat
  for (double v : kSeq) {
    ref.update(v); m.update(v);
    EXPECT_DOUBLE_EQ(m.get_median(), ref.get_median());
  }
}

TEST(SlidingMedianTest, MatchesFlatMedian_ExpectNaNTwoHeapPath) {
  FlatMedian ref(2000);
  SlidingMedian m(2000, /*expect_nan=*/true);  // > kFlatNaN(1500) -> TwoHeap
  for (double v : kSeq) {
    ref.update(v); m.update(v);
    EXPECT_DOUBLE_EQ(m.get_median(), ref.get_median());
  }
}
