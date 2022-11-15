#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/logging.hpp>
#include <ert/python.hpp>

static auto logger = ert::get_logger("enkf");

void enkf_state_initialize(enkf_fs_type *fs, enkf_node_type *param_node,
                           int iens) {
    node_id_type node_id = {.report_step = 0, .iens = iens};
    if (enkf_node_initialize(param_node, iens))
        enkf_node_store(param_node, fs, node_id);
}

ERT_CLIB_SUBMODULE("enkf_state", m) {
    m.def("state_initialize",
          [](Cwrap<enkf_node_type> param_node, Cwrap<enkf_fs_type> fs,
             int iens) { return enkf_state_initialize(fs, param_node, iens); });
}
