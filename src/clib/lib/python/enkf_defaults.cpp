#include <ert/enkf/enkf_defaults.hpp>
#include <ert/python.hpp>

ERT_CLIB_SUBMODULE("enkf_defaults", m) {
    m.add_object("DEFAULT_GEN_KW_EXPORT_NAME",
                 py::str(DEFAULT_GEN_KW_EXPORT_NAME));
}
