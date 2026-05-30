#pragma once
#include "FlatMedian.hpp"
#include "MultisetMedian.hpp"
#include "RollingMetric.hpp"
#include "TwoHeapMedian.hpp"
#include <variant>

class SlidingMedian : public RollingMetric<SlidingMedian> {
  friend class RollingMetric<SlidingMedian>;

public:
  static constexpr std::size_t kFlatClean  = 600;
  static constexpr std::size_t kFlatNaN    = 1500;
  static constexpr std::size_t kHeapClean  = 2000;

  explicit SlidingMedian(std::size_t window_size, bool expect_nan = false);
  double get_median() const;

private:
  void update_impl(double x);
  void skip_impl();
  double get_value_impl() const;
  std::size_t current_size_impl() const;

  std::variant<FlatMedian, MultisetMedian, TwoHeapMedian> impl_;
};
