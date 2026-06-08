#include "TwoHeapMedian.hpp"
#include <limits>
#include <stdexcept>

TwoHeapMedian::TwoHeapMedian(std::size_t size) : window_size_(size) {
  if (size == 0)
    throw std::invalid_argument("Window length must be greater than 0");
}

void TwoHeapMedian::clean_lower() const {
  while (!lower.empty()) {
    auto it = pending.find(lower.top());
    if (it == pending.end())
      break;
    lower.pop();
    if (--it->second == 0)
      pending.erase(it);
  }
}

void TwoHeapMedian::clean_upper() const {
  while (!upper.empty()) {
    auto it = pending.find(upper.top());
    if (it == pending.end())
      break;
    upper.pop();
    if (--it->second == 0)
      pending.erase(it);
  }
}

std::size_t TwoHeapMedian::current_size_impl() const {
  return lower_eff + upper_eff;
}

double TwoHeapMedian::get_value_impl() const {
  if (lower_eff == 0 && upper_eff == 0)
    return std::numeric_limits<double>::quiet_NaN();
  clean_lower();
  if ((lower_eff + upper_eff) % 2 == 1)
    return lower.top();
  clean_upper();
  return (lower.top() + upper.top()) / 2.0;
}

void TwoHeapMedian::evict(double oldest) {
  clean_lower();
  const bool in_lower = !lower.empty() && oldest <= lower.top();
  pending[oldest]++;
  if (in_lower) {
    lower_eff--;
    if (lower_eff < upper_eff) {
      clean_upper();
      lower.push(upper.top());
      upper.pop();
      lower_eff++;
      upper_eff--;
    }
  } else {
    upper_eff--;
    if (lower_eff > upper_eff + 1) {
      upper.push(lower.top());
      lower.pop();
      upper_eff++;
      lower_eff--;
    }
  }
}

void TwoHeapMedian::insert_value(double x) {
  clean_lower();
  if (lower.empty() || x <= lower.top()) {
    lower.push(x);
    lower_eff++;
    if (lower_eff > upper_eff + 1) {
      upper.push(lower.top());
      lower.pop();
      lower_eff--;
      upper_eff++;
    }
  } else {
    upper.push(x);
    upper_eff++;
    if (lower_eff < upper_eff) {
      clean_upper();
      lower.push(upper.top());
      upper.pop();
      upper_eff--;
      lower_eff++;
    }
  }
}

void TwoHeapMedian::update_impl(double x) {
  if (history_.size() >= window_size_) {
    double oldest = history_.front();
    history_.pop();
    if (!std::isnan(oldest))
      evict(oldest);
  }
  insert_value(x);
  history_.push(x);
}

void TwoHeapMedian::skip_impl() {
  if (history_.size() >= window_size_) {
    double oldest = history_.front();
    history_.pop();
    if (!std::isnan(oldest))
      evict(oldest);
  }
  history_.push(std::numeric_limits<double>::quiet_NaN());
}

double TwoHeapMedian::get_median() const {
  return get_value();
}
