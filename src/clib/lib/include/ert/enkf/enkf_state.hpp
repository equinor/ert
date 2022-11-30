#ifndef ERT_ENKF_STATE_H
#define ERT_ENKF_STATE_H

#include <stdbool.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.h>
#include <ert/util/stringlist.h>

#include <ert/ecl/ecl_file.h>
#include <ert/ecl/fortio.h>

#include <ert/job_queue/job_queue.hpp>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_serialize.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/ensemble_config.hpp>

std::pair<fw_load_status, std::string> enkf_state_load_from_forward_model(
    ensemble_config_type *ens_config, int last_history_restart, const int iens,
    const std::string &run_path, const std::string &job_name,
    enkf_fs_type *sim_fs);

#endif
