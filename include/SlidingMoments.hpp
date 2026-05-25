#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

class SlidingMoments {
public:
  explicit SlidingMoments(std::size_t window_size);
  void update(double x);
  void update_skewness_only(double value);
  void reset();
  std::size_t current_size() const;
  double get_skewness() const; // adjusted Fisher-Pearson
  double get_kurtosis() const; // excess kurtosis (Fisher)
  double get_mean() const;

  void process_skewness_batch(const double *in, std::size_t n, double *out,
                              std::size_t min_periods);
  void process_kurtosis_batch(const double *in, std::size_t n, double *out,
                              std::size_t min_periods);

private:
  std::size_t window_size_;
  std::vector<double> buffer_;
  std::vector<double> inv_; // inv_[n] = 1.0/n, n in [1, window_size_+1]
  std::size_t head_{0};
  std::size_t n_written_{0};
  std::size_t count_{0}; // non-NaN count in window
  double mean_{0.0};
  double M2_{0.0};
  double M3_{0.0};
  double M4_{0.0};
};
