#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <vector>

#include <ert/python.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.h>

#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_sum.h>

#include <ert/job_queue/environment_varlist.hpp>

#include "ert/enkf/ensemble_config.hpp"
#include "ert/enkf/model_config.hpp"
#include "ert/enkf/run_arg_type.hpp"

#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/gen_data.hpp>
#include <ert/logging.hpp>
#include <ert/res_util/memory.hpp>

static auto logger = ert::get_logger("enkf");

void enkf_state_initialize(enkf_fs_type *fs, enkf_node_type *param_node,
                           int iens) {
    node_id_type node_id = {.report_step = 0, .iens = iens};
    if (enkf_node_initialize(param_node, iens))
        enkf_node_store(param_node, fs, node_id);
}

bool enkf_state_complete_forward_model_EXIT_handler__(run_arg_type *run_arg) {
    if (run_arg_get_run_status(run_arg) != JOB_LOAD_FAILURE)
        run_arg_set_run_status(run_arg, JOB_RUN_FAILURE);

    auto &state_map = enkf_fs_get_state_map(run_arg_get_sim_fs(run_arg));
    state_map.set(run_arg_get_iens(run_arg), STATE_LOAD_FAILURE);
    return false;
}

ERT_CLIB_SUBMODULE("enkf_state", m) {
    m.def("state_initialize",
          [](Cwrap<enkf_node_type> param_node, Cwrap<enkf_fs_type> fs,
             int iens) { return enkf_state_initialize(fs, param_node, iens); });
}
