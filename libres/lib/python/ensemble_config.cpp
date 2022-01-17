#include <ert/enkf/ensemble_config.hpp>
#include <ert/python.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

RES_LIB_SUBMODULE("ensemble_config", m) {
    m.def(
        "ensemble_config_keylist_from_var_type",
        [](py::object self, int var_mask) {
            auto ensemble_config = py_conversion::to_ensemble_config_type(self);
            return ensemble_config_keylist_from_var_type(ensemble_config,
                                                         var_mask);
        },
        py::arg("self"), py::arg("var_mask"));
}
