#pragma once
#include "RollingMetric.hpp"
#include <queue>
#include <stdexcept>
#include <unordered_map>

class TwoHeapMedian : public RollingMetric<TwoHeapMedian> {
  friend class RollingMetric<TwoHeapMedian>;

public:
  explicit TwoHeapMedian(std::size_t window_size);

  double get_median() const;

private:
  void update_impl(double new_value);
  void skip_impl();
  double get_value_impl() const;
  std::size_t current_size_impl() const;
  void evict(double oldest);
  void clean_lower() const;
  void clean_upper() const;
  void insert_value(double x);

  std::size_t window_size_;
  mutable std::priority_queue<double> lower;
  mutable std::priority_queue<double, std::vector<double>, std::greater<double>>
      upper;
  mutable std::unordered_map<double, int> pending;
  mutable std::size_t lower_eff{0}, upper_eff{0};
  std::queue<double> history_;
};
