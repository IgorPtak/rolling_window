cmake_minimum_required(VERSION 3.14)
project(robust_rolling VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")
endif()

add_library(rolling_core
    src_core/FlatMedian.cpp
    src_core/MultisetMedian.cpp
    src_core/SlidingMedian.cpp
    src_core/TwoHeapMedian.cpp
    src_core/MonotonicMax.cpp
    src_core/MonotonicMin.cpp
    src_core/SlidingWelford.cpp
    src_core/SlidingWelfordRing.cpp
    src_core/SlidingMean.cpp
    src_core/SlidingMoments.cpp
    src_core/SlidingCovariance.cpp
)
target_include_directories(rolling_core PUBLIC include)

option(BUILD_R_WRAPPER "Build R wrapper shared library" ON)

if(BUILD_R_WRAPPER)
  find_program(R_EXECUTABLE NAMES R REQUIRED)

  execute_process(
      COMMAND ${R_EXECUTABLE} CMD config --cppflags
      OUTPUT_VARIABLE R_CPPFLAGS_RAW
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
      COMMAND ${R_EXECUTABLE} CMD config --ldflags
      OUTPUT_VARIABLE R_LDFLAGS_RAW
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  separate_arguments(R_CPPFLAGS NATIVE_COMMAND "${R_CPPFLAGS_RAW}")
  separate_arguments(R_LDFLAGS NATIVE_COMMAND "${R_LDFLAGS_RAW}")

  add_library(rolling_r_wrapper SHARED
      src/r_wrapper.cpp
  )
  target_include_directories(rolling_r_wrapper PRIVATE include)
  target_link_libraries(rolling_r_wrapper PRIVATE rolling_core)
  target_compile_options(rolling_r_wrapper PRIVATE ${R_CPPFLAGS})
  target_link_options(rolling_r_wrapper PRIVATE ${R_LDFLAGS})
endif()

include(FetchContent)

# google tests 
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)

#google benchmark
FetchContent_Declare(
    googlebenchmark 
    GIT_REPOSITORY https://github.com/google/benchmark.git 
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE) 
FetchContent_MakeAvailable(googlebenchmark)

# Required for Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing() 
add_executable(test_core
    tests/test_multiset_median.cxx
    tests/test_monotonic_max.cxx
    tests/test_monotonic_min.cxx
    tests/test_sliding_welford.cxx
    tests/test_sliding_welford_ring.cxx
    tests/test_sliding_moments.cxx
    tests/test_sliding_covariance.cxx
    tests/test_flat_median.cxx
    tests/test_two_heap_median.cxx
    tests/test_sliding_median.cxx
)
target_link_libraries(test_core PRIVATE rolling_core gtest gtest_main) 

include(GoogleTest)
gtest_discover_tests(test_core)

add_executable(bench_core benchmarks/bench_core.cxx)
target_link_libraries(bench_core PRIVATE rolling_core benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_median_sweep benchmarks/bench_median_sweep.cxx)
target_link_libraries(bench_median_sweep PRIVATE rolling_core benchmark::benchmark benchmark::benchmark_main)