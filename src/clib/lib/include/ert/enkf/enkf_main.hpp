/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_main.h' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#ifndef ERT_ENKF_MAIN_H
#define ERT_ENKF_MAIN_H

#include <string>
#include <vector>

#include <stdbool.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/res_util/ui_return.hpp>
#include <ert/util/bool_vector.h>
#include <ert/util/int_vector.h>
#include <ert/util/stringlist.h>
#include <ert/util/util.h>

#include <ert/config/config_settings.hpp>

#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/forward_model.hpp>
#include <ert/job_queue/job_queue.hpp>

#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/enkf_plot_data.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/field_config.hpp>
#include <ert/enkf/hook_manager.hpp>
#include <ert/enkf/misfit_ensemble.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/res_config.hpp>
#include <ert/enkf/site_config.hpp>

typedef struct enkf_main_struct enkf_main_type;

extern "C" void enkf_main_free(enkf_main_type *);

extern "C" enkf_main_type *enkf_main_alloc(const res_config_type *,
                                           bool read_only = false);

extern "C" enkf_obs_type *enkf_main_get_obs(const enkf_main_type *);
extern "C" bool enkf_main_have_obs(const enkf_main_type *enkf_main);

void enkf_main_install_SIGNALS(void);

extern "C" const hook_manager_type *
enkf_main_get_hook_manager(const enkf_main_type *enkf_main);

extern "C" ert_workflow_list_type *
enkf_main_get_workflow_list(enkf_main_type *enkf_main);

int enkf_main_load_from_run_context(const int ens_size,
                                    ensemble_config_type *ensemble_config,
                                    model_config_type *model_config,
                                    ecl_config_type *ecl_config,
                                    std::vector<bool> active_mask,
                                    enkf_fs_type *sim_fs,
                                    std::vector<run_arg_type *> run_args);

UTIL_SAFE_CAST_HEADER(enkf_main);
UTIL_IS_INSTANCE_HEADER(enkf_main);

#endif
