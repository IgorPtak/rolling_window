#include "MonotonicMax.hpp"
#include "MonotonicMin.hpp"
#include "MultisetMedian.hpp"
#include "SlidingCovariance.hpp"
#include "SlidingMean.hpp"
#include "SlidingMoments.hpp"
#include "SlidingWelfordRing.hpp"
#include "TwoHeapMedian.hpp"
#include <benchmark/benchmark.h>
#include <cstddef>
#include <random>
#include <vector>

static std::vector<double> make_data(std::size_t n, double nan_frac = 0.0) {
  std::mt19937 gen(42);
  std::normal_distribution<double> dist(100.0, 5.0);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  std::vector<double> v(n);
  for (auto &x : v)
    x = (coin(gen) < nan_frac) ? std::numeric_limits<double>::quiet_NaN()
                               : dist(gen);
  return v;
}

const auto DATA_CLEAN = make_data(1'000'000);
const auto DATA_NAN = make_data(1'000'000, 0.15); // 15% NaN

static void BM_MonotonicMax(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    MonotonicMax engine(w);
    for (double v : DATA_CLEAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_max());
    }
  }
}
BENCHMARK(BM_MonotonicMax)->Arg(10)->Arg(100)->Arg(1000);

static void BM_MonotonicMin(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    MonotonicMin engine(w);
    for (double v : DATA_CLEAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_min());
    }
  }
}
BENCHMARK(BM_MonotonicMin)->Arg(10)->Arg(100)->Arg(1000);

static void BM_MultisetMedian(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    MultisetMedian engine(w);
    for (double v : DATA_CLEAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_median());
    }
  }
}
BENCHMARK(BM_MultisetMedian)->Arg(10)->Arg(100)->Arg(1000)->Arg(5000)->Arg(20000);

static void BM_TwoHeapMedian(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    TwoHeapMedian engine(w);
    for (double v : DATA_CLEAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_median());
    }
  }
}
BENCHMARK(BM_TwoHeapMedian)->Arg(10)->Arg(100)->Arg(1000)->Arg(5000)->Arg(20000);

static void BM_SlidingMean(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    SlidingMean engine(w);
    for (double v : DATA_CLEAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_mean());
    }
  }
}
BENCHMARK(BM_SlidingMean)->Arg(10)->Arg(100)->Arg(1000);

static void BM_SlidingWelfordRing(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    SlidingWelfordRing engine(w);
    for (double v : DATA_CLEAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_variance());
    }
  }
}
BENCHMARK(BM_SlidingWelfordRing)->Arg(10)->Arg(100)->Arg(1000);

static void BM_SlidingMoments(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    SlidingMoments engine(w);
    for (double v : DATA_CLEAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_skewness());
      benchmark::DoNotOptimize(engine.get_kurtosis());
    }
  }
}
BENCHMARK(BM_SlidingMoments)->Arg(10)->Arg(100)->Arg(1000);

static std::pair<std::vector<double>, std::vector<double>>
make_pair_data(std::size_t n) {
  std::mt19937 gen(99);
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> x(n), y(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = dist(gen);
    y[i] = dist(gen);
  }
  return {x, y};
}

const auto [COV_X, COV_Y] = make_pair_data(100'000);

static void BM_SlidingCovariance(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    SlidingCovariance engine(w);
    for (std::size_t i = 0; i < COV_X.size(); ++i) {
      engine.update(COV_X[i], COV_Y[i]);
      benchmark::DoNotOptimize(engine.get_covariance());
    }
  }
}
BENCHMARK(BM_SlidingCovariance)->Arg(10)->Arg(100)->Arg(1000);

static void BM_MultisetMedian_NaN(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    MultisetMedian engine(w);
    for (double v : DATA_NAN) {
      if (std::isnan(v))
        engine.skip();
      else
        engine.update(v);
      benchmark::DoNotOptimize(engine.get_median());
    }
  }
}
BENCHMARK(BM_MultisetMedian_NaN)->Arg(100);

static void BM_SlidingMoments_NaN(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    SlidingMoments engine(w);
    for (double v : DATA_NAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_skewness());
    }
  }
}
BENCHMARK(BM_SlidingMoments_NaN)->Arg(100);
