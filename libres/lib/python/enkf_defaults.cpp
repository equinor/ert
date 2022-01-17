#include <ert/enkf/enkf_defaults.hpp>
#include <ert/python.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void enkf_defaults(py::module_ m) {
    m.add_object("DEFAULT_GEN_KW_EXPORT_NAME",
                 py::str(DEFAULT_GEN_KW_EXPORT_NAME));
}
