#include <ert/enkf/ensemble_config.hpp>
#include <ert/python.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

ERT_CLIB_SUBMODULE("ensemble_config", m) {
    m.def(
        "ensemble_config_keylist_from_var_type",
        [](Cwrap<ensemble_config_type> ensemble_config, int var_mask) {
            return ensemble_config_keylist_from_var_type(ensemble_config,
                                                         var_mask);
        },
        py::arg("self"), py::arg("var_mask"));
}
