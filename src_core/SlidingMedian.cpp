#include "SlidingMedian.hpp"

using Impl = std::variant<FlatMedian, MultisetMedian, TwoHeapMedian>;

static Impl make_impl(std::size_t size, bool expect_nan) {
  if (expect_nan) {
    if (size <= SlidingMedian::kFlatNaN)
      return Impl(std::in_place_index<0>, size);
    return Impl(std::in_place_index<2>, size);
  }
  if (size <= SlidingMedian::kFlatClean)
    return Impl(std::in_place_index<0>, size);
  if (size <= SlidingMedian::kHeapClean)
    return Impl(std::in_place_index<1>, size);
  return Impl(std::in_place_index<2>, size);
}

SlidingMedian::SlidingMedian(std::size_t size, bool expect_nan)
    : impl_(make_impl(size, expect_nan)) {
}

double SlidingMedian::get_median() const {
  return get_value();
}

void SlidingMedian::update_impl(double x) {
  std::visit([x](auto &m) { m.update(x); }, impl_);
}

void SlidingMedian::skip_impl() {
  std::visit([](auto &m) { m.skip(); }, impl_);
}

double SlidingMedian::get_value_impl() const {
  return std::visit([](const auto &m) { return m.get_value(); }, impl_);
}

std::size_t SlidingMedian::current_size_impl() const {
  return std::visit([](const auto &m) { return m.current_size(); }, impl_);
}
