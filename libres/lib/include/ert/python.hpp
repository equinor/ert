#ifndef RES_PYTHON_HPP
#define RES_PYTHON_HPP

#include <ert/enkf/enkf_main.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ert/logging.hpp>

namespace py = pybind11;
/**
 * This header contains utilities for interacting with Python via pybind11
 */
#endif

/**
 * Create a namespace for converting cwrap`ed python objects to c++
 */
namespace py_conversion {

enkf_main_type *to_enkf_main_type(py::object obj);
ert_run_context_type *to_run_context_type(py::object obj);
enkf_fs_type *to_enkf_fs_type(py::object obj);
ensemble_config_type *to_ensemble_config_type(py::object obj);

} // namespace py_conversion
