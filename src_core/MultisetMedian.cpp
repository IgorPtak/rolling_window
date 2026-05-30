#include "MultisetMedian.hpp"
#include <iterator>
#include <limits>
#include <stdexcept>

MultisetMedian::MultisetMedian(std::size_t size) : window_size_(size) {
  if (size <= 0) {
    throw std::invalid_argument("Window length must be greater than 0");
  }
}

bool MultisetMedian::is_even() const {
  return window_data_.size() % 2 == 0;
}

std::size_t MultisetMedian::current_size_impl() const {
  return window_data_.size();
}

double MultisetMedian::get_value_impl() const {
  if (window_data_.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (is_even()) {
    auto prev_mid_ = std::prev(mid_);
    return (*mid_ + *prev_mid_) / 2;
  } else {
    return *mid_;
  }
}

double MultisetMedian::get_median() const {
  return get_value();
}

// Recompute mid_ from scratch - used after size-changing operations.
void MultisetMedian::recompute_mid() {
  mid_ = window_data_.begin();
  std::advance(mid_, window_data_.size() / 2);
}

void MultisetMedian::skip_impl() {
  const double nan_val = std::numeric_limits<double>::quiet_NaN();
  if (history_.size() == window_size_) {
    double oldest = history_.front();
    history_.pop();
    if (!std::isnan(oldest)) {
      window_data_.erase(window_data_.lower_bound(oldest));
      if (!window_data_.empty()) {
        recompute_mid();
      }
    }
    // If oldest was NaN sentinel: nothing to evict from multiset
  }
  history_.push(nan_val);
}

void MultisetMedian::update_impl(double new_value) {
  if (history_.size() >= window_size_) {
    double oldest_val = history_.front();
    history_.pop();
    history_.push(new_value);
    window_data_.insert(new_value);

    if (!std::isnan(oldest_val)) {
      // Normal evict+insert: size stays same use incremental mid_ adjustment.
      if (new_value < *mid_) {
        mid_--;
      }
      auto it_to_remove = window_data_.lower_bound(oldest_val);
      if (oldest_val <= *mid_) {
        mid_++;
      }
      window_data_.erase(it_to_remove);
    } else {
      // Oldest was a NaN sentinel: insert without eviction, recompute mid_.
      recompute_mid();
    }
  } else {
    history_.push(new_value);
    window_data_.insert(new_value);

    if (window_data_.size() == 1) {
      mid_ = window_data_.begin();
    } else {
      if (new_value < *mid_) {
        mid_--;
      }
      if (window_data_.size() % 2 == 0) {
        mid_++;
      }
    }
  }
}
