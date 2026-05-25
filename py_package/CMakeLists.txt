cmake_minimum_required(VERSION 3.15)
project(robust_rolling_python LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(robust_rolling_core py_wrapper.cpp)

target_include_directories(robust_rolling_core PRIVATE ../include)

target_sources(robust_rolling_core PRIVATE
  ../src_core/MonotonicMax.cpp
  ../src_core/MonotonicMin.cpp
  ../src_core/MultisetMedian.cpp
  ../src_core/SlidingCovariance.cpp
  ../src_core/SlidingMean.cpp
  ../src_core/SlidingMoments.cpp
  ../src_core/SlidingMomentsPrefix.cpp
  ../src_core/SlidingWelfordRing.cpp
)

if(MSVC)
  target_compile_options(robust_rolling_core PRIVATE /O2 /DNDEBUG)
else()
  target_compile_options(robust_rolling_core PRIVATE -O3 -march=native -funroll-loops -ffp-contract=fast -DNDEBUG -flto)
  target_link_options(robust_rolling_core PRIVATE -flto)
endif()

install(TARGETS robust_rolling_core LIBRARY DESTINATION .)