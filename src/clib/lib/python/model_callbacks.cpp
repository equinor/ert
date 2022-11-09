#include <ert/enkf/enkf_types.hpp>
#include <ert/python.hpp>

ERT_CLIB_SUBMODULE("model_callbacks", m) {
    py::enum_<fw_load_status>(m, "LoadStatus", py::arithmetic())
        .value("LOAD_SUCCESSFUL", LOAD_SUCCESSFUL)
        .value("LOAD_FAILURE", LOAD_FAILURE)
        .value("TIME_MAP_FAILURE", TIME_MAP_FAILURE)
        .export_values();
}
