#include "../include/MonotonicMax.hpp"
#include "../include/MonotonicMin.hpp"
#include "../include/MultisetMedian.hpp"
#include "../include/SlidingCovariance.hpp"
#include "../include/SlidingMean.hpp"
#include "../include/SlidingMoments.hpp"
#include "../include/SlidingMomentsPrefix.hpp"
#include "../include/SlidingWelfordRing.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename Metric, typename Getter>
py::array_t<double> process_batch_generic(
    Metric &metric,
    py::array_t<double, py::array::c_style | py::array::forcecast> input_array,
    Getter getter, std::size_t min_periods) {

  py::buffer_info in_info = input_array.request();
  if (in_info.ndim != 1)
    throw std::runtime_error("Input must be 1D array");

  auto result_array = py::array_t<double>(
      py::array::ShapeContainer{in_info.shape[0]},
      py::array::StridesContainer{static_cast<py::ssize_t>(sizeof(double))});
  py::buffer_info out_info = result_array.request();

  const auto *in_ptr = static_cast<const double *>(in_info.ptr);
  auto *out_ptr = static_cast<double *>(out_info.ptr);

  for (py::ssize_t i = 0; i < in_info.shape[0]; ++i) {
    const double value = in_ptr[i];
    if (std::isnan(value)) {
      metric.skip();
      out_ptr[i] = getter(metric);
    } else {
      metric.update(value);
      out_ptr[i] = getter(metric);
    }

    if (metric.current_size() < min_periods)
      out_ptr[i] = std::numeric_limits<double>::quiet_NaN();
  }

  return result_array;
}

PYBIND11_MODULE(robust_rolling_core, m) {
  m.doc() = "Rolling Metrics Engine (Python Bindings)";

  py::class_<SlidingWelfordRing>(m, "SlidingWelford")
      .def(py::init<std::size_t>())
      .def("update", &SlidingWelfordRing::update)
      .def("get_variance", &SlidingWelfordRing::get_value)
      .def(
          "process_batch",
          [](SlidingWelfordRing &self,
             py::array_t<double, py::array::c_style | py::array::forcecast>
                 input,
             std::size_t min_periods) {
            return process_batch_generic(
                self, input,
                [](SlidingWelfordRing &m) { return m.get_variance(); },
                min_periods);
          },
          py::arg("input"), py::arg("min_periods") = 0);

  py::class_<MonotonicMax>(m, "MonotonicMax")
      .def(py::init<std::size_t>())
      .def("update", &MonotonicMax::update)
      .def("get_max", &MonotonicMax::get_max)
      .def(
          "process_batch",
          [](MonotonicMax &self,
             py::array_t<double, py::array::c_style | py::array::forcecast>
                 input,
             std::size_t min_periods) {
            return process_batch_generic(
                self, input, [](MonotonicMax &m) { return m.get_max(); },
                min_periods);
          },
          py::arg("input"), py::arg("min_periods") = 0);

  py::class_<MonotonicMin>(m, "MonotonicMin")
      .def(py::init<std::size_t>())
      .def("update", &MonotonicMin::update)
      .def("get_min", &MonotonicMin::get_min)
      .def(
          "process_batch",
          [](MonotonicMin &self,
             py::array_t<double, py::array::c_style | py::array::forcecast>
                 input,
             std::size_t min_periods) {
            return process_batch_generic(
                self, input, [](MonotonicMin &m) { return m.get_min(); },
                min_periods);
          },
          py::arg("input"), py::arg("min_periods") = 0);

  py::class_<MultisetMedian>(m, "MultisetMedian")
      .def(py::init<std::size_t>())
      .def("update", &MultisetMedian::update)
      .def("get_median", &MultisetMedian::get_median)
      .def(
          "process_batch",
          [](MultisetMedian &self,
             py::array_t<double, py::array::c_style | py::array::forcecast>
                 input,
             std::size_t min_periods) {
            return process_batch_generic(
                self, input, [](MultisetMedian &m) { return m.get_median(); },
                min_periods);
          },
          py::arg("input"), py::arg("min_periods") = 0);

  py::class_<SlidingMean>(m, "SlidingMean")
      .def(py::init<std::size_t>())
      .def("update", &SlidingMean::update)
      .def("get_mean", &SlidingMean::get_mean)
      .def(
          "process_batch",
          [](SlidingMean &self,
             py::array_t<double, py::array::c_style | py::array::forcecast>
                 input,
             std::size_t min_periods, bool assume_finite) {
            py::buffer_info info = input.request();
            if (info.ndim != 1)
              throw std::runtime_error("Input must be 1D array");
            std::size_t n = static_cast<std::size_t>(info.shape[0]);
            auto result = py::array_t<double>(
                py::array::ShapeContainer{info.shape[0]},
                py::array::StridesContainer{
                    static_cast<py::ssize_t>(sizeof(double))});
            const double *in_ptr = static_cast<const double *>(info.ptr);
            double *out_ptr = static_cast<double *>(result.request().ptr);
            if (assume_finite)
              self.fast_mean_batch_finite(in_ptr, n, out_ptr, min_periods);
            else
              self.fast_mean_batch(in_ptr, n, out_ptr, min_periods);
            return result;
          },
          py::arg("input"), py::arg("min_periods") = 0,
          py::arg("assume_finite") = false);

  py::class_<SlidingMoments>(m, "SlidingMoments")
      .def(py::init<std::size_t>())
      .def("update", &SlidingMoments::update)
      .def("reset", &SlidingMoments::reset)
      .def("current_size", &SlidingMoments::current_size)
      .def("get_mean", &SlidingMoments::get_mean)
      .def("get_skewness", &SlidingMoments::get_skewness)
      .def("get_kurtosis", &SlidingMoments::get_kurtosis)
      .def("process_mean_batch",
           [](SlidingMoments &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  input,
              std::size_t min_periods) {
             // min_periods=0 means no restriction
             py::buffer_info info = input.request();
             if (info.ndim != 1)
               throw std::runtime_error("Input must be 1D array");
             auto result = py::array_t<double>(
                 py::array::ShapeContainer{info.shape[0]},
                 py::array::StridesContainer{
                     static_cast<py::ssize_t>(sizeof(double))});
             const auto *in = static_cast<const double *>(info.ptr);
             auto *out = static_cast<double *>(result.request().ptr);
             for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
               self.update(in[i]);
               out[i] = self.current_size() < min_periods
                            ? std::numeric_limits<double>::quiet_NaN()
                            : self.get_mean();
             }
             return result;
           })
      .def("process_skewness_batch",
           [](SlidingMoments &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  input,
              std::size_t min_periods) {
             py::buffer_info info = input.request();
             if (info.ndim != 1)
               throw std::runtime_error("Input must be 1D array");
             auto result = py::array_t<double>(
                 py::array::ShapeContainer{info.shape[0]},
                 py::array::StridesContainer{
                     static_cast<py::ssize_t>(sizeof(double))});
             const auto *in = static_cast<const double *>(info.ptr);
             auto *out = static_cast<double *>(result.request().ptr);
             self.process_skewness_batch(
                 in, static_cast<std::size_t>(info.shape[0]), out, min_periods);
             return result;
           })
      .def("process_kurtosis_batch",
           [](SlidingMoments &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  input,
              std::size_t min_periods) {
             py::buffer_info info = input.request();
             if (info.ndim != 1)
               throw std::runtime_error("Input must be 1D array");
             auto result = py::array_t<double>(
                 py::array::ShapeContainer{info.shape[0]},
                 py::array::StridesContainer{
                     static_cast<py::ssize_t>(sizeof(double))});
             const auto *in = static_cast<const double *>(info.ptr);
             auto *out = static_cast<double *>(result.request().ptr);
             self.process_kurtosis_batch(
                 in, static_cast<std::size_t>(info.shape[0]), out, min_periods);
             return result;
           });

  py::class_<SlidingCovariance>(m, "SlidingCovariance")
      .def(py::init<std::size_t>())
      .def("update", &SlidingCovariance::update)
      .def("get_covariance", &SlidingCovariance::get_covariance)
      .def("get_correlation", &SlidingCovariance::get_correlation)
      .def("get_mean_x", &SlidingCovariance::get_mean_x)
      .def("get_mean_y", &SlidingCovariance::get_mean_y)
      .def(
          "process_covariance_batch",
          [](SlidingCovariance &self,
             py::array_t<double, py::array::c_style | py::array::forcecast> x,
             py::array_t<double, py::array::c_style | py::array::forcecast> y) {
            py::buffer_info xi = x.request(), yi = y.request();
            if (xi.ndim != 1 || yi.ndim != 1)
              throw std::runtime_error("Inputs must be 1D arrays");
            if (xi.shape[0] != yi.shape[0])
              throw std::runtime_error("x and y must have the same length");
            auto result = py::array_t<double>(
                py::array::ShapeContainer{xi.shape[0]},
                py::array::StridesContainer{
                    static_cast<py::ssize_t>(sizeof(double))});
            const auto *xp = static_cast<const double *>(xi.ptr);
            const auto *yp = static_cast<const double *>(yi.ptr);
            auto *out = static_cast<double *>(result.request().ptr);
            for (py::ssize_t i = 0; i < xi.shape[0]; ++i) {
              self.update(xp[i], yp[i]);
              out[i] = self.get_covariance();
            }
            return result;
          })
      .def(
          "process_correlation_batch",
          [](SlidingCovariance &self,
             py::array_t<double, py::array::c_style | py::array::forcecast> x,
             py::array_t<double, py::array::c_style | py::array::forcecast> y) {
            py::buffer_info xi = x.request(), yi = y.request();
            if (xi.ndim != 1 || yi.ndim != 1)
              throw std::runtime_error("Inputs must be 1D arrays");
            if (xi.shape[0] != yi.shape[0])
              throw std::runtime_error("x and y must have the same length");
            auto result = py::array_t<double>(
                py::array::ShapeContainer{xi.shape[0]},
                py::array::StridesContainer{
                    static_cast<py::ssize_t>(sizeof(double))});
            const auto *xp = static_cast<const double *>(xi.ptr);
            const auto *yp = static_cast<const double *>(yi.ptr);
            auto *out = static_cast<double *>(result.request().ptr);
            for (py::ssize_t i = 0; i < xi.shape[0]; ++i) {
              self.update(xp[i], yp[i]);
              out[i] = self.get_correlation();
            }
            return result;
          });

  auto make_prefix_batch = [&](auto method_ptr) {
    return [method_ptr](
               SlidingMomentsPrefix &self,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   input,
               std::size_t min_periods) {
      py::buffer_info info = input.request();
      if (info.ndim != 1)
        throw std::runtime_error("Input must be 1D array");
      std::size_t n = static_cast<std::size_t>(info.shape[0]);
      auto result =
          py::array_t<double>(py::array::ShapeContainer{info.shape[0]},
                              py::array::StridesContainer{
                                  static_cast<py::ssize_t>(sizeof(double))});
      (self.*method_ptr)(static_cast<const double *>(info.ptr), n,
                         static_cast<double *>(result.request().ptr),
                         min_periods);
      return result;
    };
  };

  py::class_<SlidingMomentsPrefix>(m, "SlidingMomentsPrefix")
      .def(py::init<std::size_t>())
      .def("variance_batch",
           make_prefix_batch(&SlidingMomentsPrefix::variance_batch),
           py::arg("input"), py::arg("min_periods") = 0)
      .def("skewness_batch",
           make_prefix_batch(&SlidingMomentsPrefix::skewness_batch),
           py::arg("input"), py::arg("min_periods") = 0)
      .def("kurtosis_batch",
           make_prefix_batch(&SlidingMomentsPrefix::kurtosis_batch),
           py::arg("input"), py::arg("min_periods") = 0);
}
