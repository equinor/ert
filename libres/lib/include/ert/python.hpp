/**
 * This header contains utilities for interacting with Python via pybind11
 */

#pragma once

#include <ert/enkf/enkf_main.hpp>
#include <ert/enkf/enkf_plot_gendata.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <ert/logging.hpp>

namespace py = pybind11;

namespace ert::detail {
struct Submodule {
    using init_type = void(py::module_);
    const char *path;
    init_type &init;

    Submodule(const char *, init_type &);
};
} // namespace ert::detail

/**
 * Define a submodule path within the Python package 'res._lib'
 *
 * This is macro is similar to Pybind11's PYBIND11_MODULE macro. The first
 * argument is the Python submodule path, and the second is the name of the
 * py::module_ parameter (eg. 'm').
 *
 * For example, the following will create the Python module 'res._lib.foo.bar'
 * which contains an object named 'baz' whose value is the string 'quz'.
 *
 *     RES_LIB_SUBMODULE("foo.bar", m) {
 *         m.add_object("baz", py::str{"quz"});
 *     }
 *
 * Multiple initialisation functions can be defined for the same submodule.
 * However, the order in which each function is called is undefined.
 *
 * Note: The name of this macro should reflect the module path of this libres
 * library. At the moment it is 'res._lib', so the macro is prefixed with
 * RES_LIB.
 */
#define RES_LIB_SUBMODULE(_Path, _ModuleParam)                                 \
    static void _python_submodule_init(py::module_);                           \
    static ::ert::detail::Submodule _python_submodule{_Path,                   \
                                                      _python_submodule_init}; \
    void _python_submodule_init(py::module_ _ModuleParam)

namespace py_conversion {

enkf_main_type *to_enkf_main_type(py::object obj);
ert_run_context_type *to_run_context_type(py::object obj);
enkf_fs_type *to_enkf_fs_type(py::object obj);
ensemble_config_type *to_ensemble_config_type(py::object obj);
enkf_plot_gendata_type *to_enkf_plot_gendata_type(py::object obj);

} // namespace py_conversion
