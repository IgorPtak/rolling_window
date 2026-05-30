#pragma once
#include "RollingMetric.hpp"
#include <algorithm>
#include <queue>
#include <vector>

class FlatMedian : public RollingMetric<FlatMedian> {
  friend class RollingMetric<FlatMedian>;

public:
  explicit FlatMedian(std::size_t window_size);
  double get_median() const;

private:
  void update_impl(double x);
  void skip_impl();
  double get_value_impl() const;
  std::size_t current_size_impl() const;

  std::size_t window_size_;
  std::vector<double> sorted_;
  std::queue<double> history_;
};
