#include "TwoHeapMedian.hpp"
#include <iterator>
#include <limits>
#include <stdexcept>

TwoHeapMedian::TwoHeapMedian(std::size_t size) : window_size_(size) {
  if (size <= 0) {
    throw std::invalid_argument("Window length must be greater than 0");
  }
}

void TwoHeapMedian::clean_lower() const {
  while (!lower.empty() && pending.count(lower.top()) > 0) {
    pending[lower.top()]--;
    lower.pop();
    lower_eff--;
  }
}

void TwoHeapMedian::clean_upper() const {
  while (!upper.empty() && pending.count(upper.top()) > 0) {
    pending[upper.top()]--;
    upper.pop();
    upper_eff--;
  }
}

std::size_t TwoHeapMedian::current_size_impl() const {
  return lower_eff + upper_eff;
}

double TwoHeapMedian::get_value_impl() const {
  clean_lower();
  clean_upper();

  if (lower.empty() && upper.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (lower_eff > upper_eff) {
    return lower.top();
  } else {
    return (lower.top() + upper.top()) / 2;
  }
}

void TwoHeapMedian::evict(double oldest) {
  pending[oldest]++;

  clean_lower();

  if (!lower.empty() && oldest <= lower.top()) {
    lower_eff--;
  } else {
    upper_eff--;
  }

  if (lower_eff > upper_eff + 1) {
    clean_lower();
    upper.push(lower.top());
    lower.pop();
    lower_eff--;
    upper_eff++;
  } else if (upper_eff > lower_eff) {
    clean_upper();
    lower.push(upper.top());
    upper.pop();
    upper_eff--;
    lower_eff++;
  }
}

void TwoHeapMedian::insert_value(double x) {
  if (lower.empty() || x <= lower.top()) {
    lower.push(x);
    lower_eff++;
  } else {
    upper.push(x);
    upper_eff++;
  }

  if (lower_eff > upper_eff + 1) {
    upper.push(lower.top());
    lower.pop();
    lower_eff--;
    upper_eff++;
  } else if (upper_eff > lower_eff) {
    lower.push(upper.top());
    upper.pop();
    upper_eff--;
    lower_eff++;
  }
}

void TwoHeapMedian::update_impl(double x) {
  if (history_.size() >= window_size_) {
    double oldest = history_.front();
    history_.pop();
    if (!std::isnan(oldest)) {
      evict(oldest);
    }
  }
  insert_value(x);
  history_.push(x);
}

void TwoHeapMedian::skip_impl() {
  if (history_.size() >= window_size_) {
    double oldest = history_.front();
    history_.pop();
    if (!std::isnan(oldest)) {
      evict(oldest);
    }
  }
  history_.push(std::numeric_limits<double>::quiet_NaN());
}

double TwoHeapMedian::get_median() const {
  return get_value();
}