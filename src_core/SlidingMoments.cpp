#include "SlidingMoments.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

SlidingMoments::SlidingMoments(std::size_t window_size)
    : window_size_(window_size) {
  if (window_size_ == 0)
    throw std::invalid_argument("Window length must be greater than 0");

  buffer_.resize(window_size_, kNaN);

  // Inverse table: inv_[n] = 1.0/n for n in [1, window_size_+1].
  // Eliminates all divisions in the hot path.
  inv_.resize(window_size_ + 2, 0.0);
  for (std::size_t j = 1; j <= window_size_ + 1; ++j)
    inv_[j] = 1.0 / static_cast<double>(j);
}

static inline void
moments_add(double x, std::size_t &cnt, double &mean, double &M2, double &M3,
            double &M4, const double *inv) {
  ++cnt;
  double n = static_cast<double>(cnt);
  double d = x - mean;
  double dn = d * inv[cnt];
  double t = d * dn * (n - 1.0);
  M4 += t * dn * dn * (n * n - 3.0 * n + 3.0) + 6.0 * dn * dn * M2 -
        4.0 * dn * M3;
  M3 += t * dn * (n - 2.0) - 3.0 * dn * M2;
  M2 += t;
  mean += dn;
}

static inline void
moments_remove(double x_out, std::size_t &cnt, double &mean, double &M2,
               double &M3, double &M4, const double *inv) {
  if (cnt == 1) {
    mean = M2 = M3 = M4 = 0.0;
    cnt = 0;
    return;
  }
  double n = static_cast<double>(cnt);
  double mn = (n * mean - x_out) * inv[cnt - 1];
  double d = x_out - mn;
  double dn = d * inv[cnt];
  double t = d * dn * (n - 1.0);
  double M2o = M2 - t;
  double M3o = M3 - (t * dn * (n - 2.0) - 3.0 * dn * M2o);
  M4 -= t * dn * dn * (n * n - 3.0 * n + 3.0) + 6.0 * dn * dn * M2o -
        4.0 * dn * M3o;
  M3 = M3o;
  M2 = M2o > 0.0 ? M2o : 0.0;
  mean = mn;
  --cnt;
}

static inline void
moments_add_3(double x, std::size_t &cnt, double &mean, double &M2, double &M3,
              const double *inv) {
  ++cnt;
  double n = static_cast<double>(cnt);
  double d = x - mean;
  double dn = d * inv[cnt];
  double t = d * dn * (n - 1.0);
  M3 += t * dn * (n - 2.0) - 3.0 * dn * M2;
  M2 += t;
  mean += dn;
}

static inline void
moments_remove_3(double x_out, std::size_t &cnt, double &mean, double &M2,
                 double &M3, const double *inv) {
  if (cnt == 1) {
    mean = M2 = M3 = 0.0;
    cnt = 0;
    return;
  }
  double n = static_cast<double>(cnt);
  double mn = (n * mean - x_out) * inv[cnt - 1];
  double d = x_out - mn;
  double dn = d * inv[cnt];
  double t = d * dn * (n - 1.0);
  double M2o = M2 - t;
  M3 -= t * dn * (n - 2.0) - 3.0 * dn * M2o;
  M2 = M2o > 0.0 ? M2o : 0.0;
  mean = mn;
  --cnt;
}

std::size_t SlidingMoments::current_size() const {
  return count_;
}

void SlidingMoments::reset() {
  buffer_.assign(window_size_, kNaN);
  head_ = n_written_ = count_ = 0;
  mean_ = M2_ = M3_ = M4_ = 0.0;
}

double SlidingMoments::get_skewness() const {
  if (count_ < 3 || M2_ <= 0.0)
    return kNaN;
  double n = static_cast<double>(count_);
  return M3_ * n * std::sqrt(n - 1.0) / (M2_ * std::sqrt(M2_) * (n - 2.0));
}

double SlidingMoments::get_kurtosis() const {
  if (count_ < 4 || M2_ <= 0.0)
    return kNaN;
  double n = static_cast<double>(count_);
  double g2 = M4_ * n / (M2_ * M2_) - 3.0;
  return (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 + 6.0);
}

double SlidingMoments::get_mean() const {
  return count_ == 0 ? kNaN : mean_;
}

void SlidingMoments::update(double value) {
  const bool ring_full = (n_written_ >= window_size_);
  if (ring_full) {
    double x_out = buffer_[head_];
    if (!std::isnan(x_out))
      moments_remove(x_out, count_, mean_, M2_, M3_, M4_, inv_.data());
  }
  buffer_[head_] = value;
  if (++head_ == window_size_)
    head_ = 0;
  if (!ring_full)
    ++n_written_;
  if (!std::isnan(value))
    moments_add(value, count_, mean_, M2_, M3_, M4_, inv_.data());
}

void SlidingMoments::update_skewness_only(double value) {
  const bool ring_full = (n_written_ >= window_size_);
  if (ring_full) {
    double x_out = buffer_[head_];
    if (!std::isnan(x_out))
      moments_remove_3(x_out, count_, mean_, M2_, M3_, inv_.data());
  }
  buffer_[head_] = value;
  if (++head_ == window_size_)
    head_ = 0;
  if (!ring_full)
    ++n_written_;
  if (!std::isnan(value))
    moments_add_3(value, count_, mean_, M2_, M3_, inv_.data());
}

void SlidingMoments::process_skewness_batch(const double *in,
                                            std::size_t n,
                                            double *out,
                                            std::size_t min_periods) {
  const std::size_t k = window_size_;
  const double ck = static_cast<double>(k);
  const double *iv = inv_.data();

  // Precomputed skewness scale for the steady-state (count_ == k) case:
  // get_skewness = M3_ * ck * sqrt(ck-1) / (M2_ * sqrt(M2_) * (ck-2))
  const double skew_scale =
      (k >= 3) ? ck * std::sqrt(ck - 1.0) / (ck - 2.0) : kNaN;

  std::size_t i = 0;
  for (; i < n && n_written_ < k; ++i) {
    buffer_[head_] = in[i];
    if (++head_ == k)
      head_ = 0;
    ++n_written_;
    if (!std::isnan(in[i]))
      moments_add_3(in[i], count_, mean_, M2_, M3_, iv);
    out[i] = count_ < min_periods ? kNaN : get_skewness();
  }

  for (; i < n; ++i) {
    double x_out = buffer_[head_];
    double x_in = in[i];
    buffer_[head_] = x_in;
    if (++head_ == k)
      head_ = 0;

    if (!std::isnan(x_out))
      moments_remove_3(x_out, count_, mean_, M2_, M3_, iv);

    if (!std::isnan(x_in))
      moments_add_3(x_in, count_, mean_, M2_, M3_, iv);

    if (count_ < min_periods) {
      out[i] = kNaN;
    } else if (count_ == k) {
      out[i] = M2_ > 0.0 ? M3_ * skew_scale / (M2_ * std::sqrt(M2_)) : kNaN;
    } else {
      out[i] = get_skewness();
    }
  }
}

void SlidingMoments::process_kurtosis_batch(const double *in,
                                            std::size_t n,
                                            double *out,
                                            std::size_t min_periods) {
  const std::size_t k = window_size_;
  const double ck = static_cast<double>(k);
  const double *iv = inv_.data();

  // Precomputed kurtosis scale for steady-state (count_ == k):
  // get_kurtosis = (n-1)/((n-2)*(n-3)) * ((n+1)*g2 + 6)
  // where g2 = M4_*n/(M2_*M2_) - 3
  // We precompute the outer coefficient and the (n+1)/((n-2)*(n-3)) part.
  const double kurt_outer =
      (k >= 4) ? (ck - 1.0) / ((ck - 2.0) * (ck - 3.0)) : kNaN;
  const double kurt_n1 = ck + 1.0;
  const double kurt_n_inv =
      ck; // multiplier for M4_/(M2_*M2_): ck*kurt_outer*(n+1)

  std::size_t i = 0;
  for (; i < n && n_written_ < k; ++i) {
    buffer_[head_] = in[i];
    if (++head_ == k)
      head_ = 0;
    ++n_written_;
    if (!std::isnan(in[i]))
      moments_add(in[i], count_, mean_, M2_, M3_, M4_, iv);
    out[i] = count_ < min_periods ? kNaN : get_kurtosis();
  }

  for (; i < n; ++i) {
    double x_out = buffer_[head_];
    double x_in = in[i];
    buffer_[head_] = x_in;
    if (++head_ == k)
      head_ = 0;

    if (!std::isnan(x_out))
      moments_remove(x_out, count_, mean_, M2_, M3_, M4_, iv);

    if (!std::isnan(x_in))
      moments_add(x_in, count_, mean_, M2_, M3_, M4_, iv);

    if (count_ < min_periods) {
      out[i] = kNaN;
    } else if (count_ == k) {
      if (M2_ <= 0.0) {
        out[i] = kNaN;
        continue;
      }
      double M2sq = M2_ * M2_;
      double g2 = M4_ * kurt_n_inv / M2sq - 3.0;
      out[i] = kurt_outer * (kurt_n1 * g2 + 6.0);
    } else {
      out[i] = get_kurtosis();
    }
  }
}
