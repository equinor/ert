#ifndef ERT_ENKF_STATE_H
#define ERT_ENKF_STATE_H

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/run_arg.hpp>

void enkf_state_initialize(enkf_fs_type *fs, enkf_node_type *param_node,
                           int iens);

bool enkf_state_complete_forward_model_EXIT_handler__(run_arg_type *run_arg);

#endif
