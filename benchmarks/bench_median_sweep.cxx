#include "FlatMedian.hpp"
#include "MultisetMedian.hpp"
#include "SlidingMedian.hpp"
#include "TwoHeapMedian.hpp"
#include <benchmark/benchmark.h>
#include <cstddef>
#include <limits>
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

const auto DATA_CLEAN = make_data(500'000);
const auto DATA_NAN   = make_data(500'000, 0.15);


template <typename Engine>
static void run_clean(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    Engine engine(w);
    for (double v : DATA_CLEAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_median());
    }
  }
}

template <typename Engine>
static void run_nan(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    Engine engine(w);
    for (double v : DATA_NAN) {
      if (std::isnan(v)) engine.skip();
      else               engine.update(v);
      benchmark::DoNotOptimize(engine.get_median());
    }
  }
}

#define WINDOW_ARGS \
  ->Arg(10)->Arg(50)->Arg(100)->Arg(200)->Arg(300)->Arg(400) \
  ->Arg(500)->Arg(600)->Arg(700)->Arg(800)->Arg(1000)->Arg(2000)->Arg(5000)

static void BM_Flat_Clean(benchmark::State &s)     { run_clean<FlatMedian>(s); }
static void BM_Multiset_Clean(benchmark::State &s) { run_clean<MultisetMedian>(s); }
static void BM_TwoHeap_Clean(benchmark::State &s)  { run_clean<TwoHeapMedian>(s); }
static void BM_Sliding_Clean(benchmark::State &s)  { run_clean<SlidingMedian>(s); }

static void BM_SlidingNaNHint_Clean(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    SlidingMedian engine(w, /*expect_nan=*/true);
    for (double v : DATA_CLEAN) {
      engine.update(v);
      benchmark::DoNotOptimize(engine.get_median());
    }
  }
}

BENCHMARK(BM_Flat_Clean)          WINDOW_ARGS;
BENCHMARK(BM_Multiset_Clean)      WINDOW_ARGS;
BENCHMARK(BM_TwoHeap_Clean)       WINDOW_ARGS;
BENCHMARK(BM_Sliding_Clean)       WINDOW_ARGS;
BENCHMARK(BM_SlidingNaNHint_Clean) WINDOW_ARGS;

static void BM_Flat_NaN(benchmark::State &s)     { run_nan<FlatMedian>(s); }
static void BM_Multiset_NaN(benchmark::State &s) { run_nan<MultisetMedian>(s); }
static void BM_TwoHeap_NaN(benchmark::State &s)  { run_nan<TwoHeapMedian>(s); }
static void BM_Sliding_NaN(benchmark::State &s)  { run_nan<SlidingMedian>(s); }

static void BM_SlidingNaNHint_NaN(benchmark::State &state) {
  std::size_t w = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    SlidingMedian engine(w, /*expect_nan=*/true);
    for (double v : DATA_NAN) {
      if (std::isnan(v)) engine.skip();
      else               engine.update(v);
      benchmark::DoNotOptimize(engine.get_median());
    }
  }
}

BENCHMARK(BM_Flat_NaN)           WINDOW_ARGS;
BENCHMARK(BM_Multiset_NaN)       WINDOW_ARGS;
BENCHMARK(BM_TwoHeap_NaN)        WINDOW_ARGS;
BENCHMARK(BM_Sliding_NaN)        WINDOW_ARGS;
BENCHMARK(BM_SlidingNaNHint_NaN) WINDOW_ARGS;
