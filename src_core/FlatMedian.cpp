#include "FlatMedian.hpp"
#include <limits>
#include <stdexcept>

FlatMedian::FlatMedian(std::size_t size) : window_size_(size) {
  if (size == 0)
    throw std::invalid_argument("Window length must be greater than 0");
  sorted_.reserve(size);
}

std::size_t FlatMedian::current_size_impl() const { return sorted_.size(); }

double FlatMedian::get_value_impl() const {
  if (sorted_.empty())
    return std::numeric_limits<double>::quiet_NaN();
  const std::size_t n = sorted_.size();
  return (n % 2 == 1) ? sorted_[n / 2]
                      : (sorted_[n / 2 - 1] + sorted_[n / 2]) / 2.0;
}

double FlatMedian::get_median() const { return get_value(); }

void FlatMedian::update_impl(double x) {
  if (history_.size() >= window_size_) {
    double oldest = history_.front();
    history_.pop();
    if (!std::isnan(oldest)) {
      auto it = std::lower_bound(sorted_.begin(), sorted_.end(), oldest);
      sorted_.erase(it);
    }
  }
  auto pos = std::lower_bound(sorted_.begin(), sorted_.end(), x);
  sorted_.insert(pos, x);
  history_.push(x);
}

void FlatMedian::skip_impl() {
  if (history_.size() >= window_size_) {
    double oldest = history_.front();
    history_.pop();
    if (!std::isnan(oldest)) {
      auto it = std::lower_bound(sorted_.begin(), sorted_.end(), oldest);
      sorted_.erase(it);
    }
  }
  history_.push(std::numeric_limits<double>::quiet_NaN());
}
