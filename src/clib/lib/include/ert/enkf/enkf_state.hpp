#ifndef ERT_ENKF_STATE_H
#define ERT_ENKF_STATE_H

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_node.hpp>

void enkf_state_initialize(enkf_fs_type *fs, enkf_node_type *param_node,
                           int iens);

#endif
