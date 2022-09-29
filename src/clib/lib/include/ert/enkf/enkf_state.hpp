#ifndef ERT_ENKF_STATE_H
#define ERT_ENKF_STATE_H

#include <stdbool.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.h>
#include <ert/util/rng.h>
#include <ert/util/stringlist.h>

#include <ert/ecl/ecl_file.h>
#include <ert/ecl/fortio.h>

#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/forward_model.hpp>
#include <ert/job_queue/job_queue.hpp>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_serialize.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/enkf/run_arg.hpp>
#include <ert/enkf/site_config.hpp>

void enkf_state_initialize(rng_type *rng, enkf_fs_type *fs,
                           enkf_node_type *param_node, int iens);

std::pair<fw_load_status, std::string>
enkf_state_load_from_forward_model(ensemble_config_type *ens_config,
                                   model_config_type *model_config,
                                   const run_arg_type *run_arg);
bool enkf_state_complete_forward_model_EXIT_handler__(run_arg_type *run_arg);

#endif
