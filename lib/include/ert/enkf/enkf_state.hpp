/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_state.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_ENKF_STATE_H
#define ERT_ENKF_STATE_H

#include <stdbool.h>

#include <ert/util/hash.h>
#include <ert/util/rng.h>
#include <ert/util/stringlist.h>
#include <ert/util/rng.h>
#include <ert/res_util/subst_list.hpp>
#include <ert/res_util/matrix.hpp>

#include <ert/ecl/fortio.h>
#include <ert/ecl/ecl_file.h>

#include <ert/job_queue/forward_model.hpp>
#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/job_queue.hpp>

#include <ert/enkf/model_config.hpp>
#include <ert/enkf/site_config.hpp>
#include <ert/enkf/ecl_config.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/res_config.hpp>
#include <ert/enkf/ert_template.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/enkf_serialize.hpp>
#include <ert/enkf/run_arg.hpp>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct enkf_state_struct    enkf_state_type;

  void *             enkf_state_load_from_forward_model_mt( void * arg );
  void               enkf_state_initialize(enkf_state_type * enkf_state , rng_type * rng, enkf_fs_type * fs, const stringlist_type * param_list , init_mode_type init_mode);

  int                enkf_state_load_from_forward_model(enkf_state_type * enkf_state ,
                                                        run_arg_type * run_arg ,
                                                        stringlist_type * msg_list);

  void enkf_state_init_eclipse(const res_config_type * res_config,
                               const run_arg_type * run_arg );

  enkf_state_type  * enkf_state_alloc(int ,
                                      rng_type        * main_rng ,
                                      model_config_type * ,
                                      ensemble_config_type * ,
                                      const site_config_type * ,
                                      const ecl_config_type * ,
                                      ert_templates_type * templates);

  void               enkf_state_add_node(enkf_state_type * , const char *  , const enkf_config_node_type * );
  void               enkf_state_ecl_write(const ensemble_config_type * ens_config, const model_config_type * model_config, const run_arg_type * run_arg , enkf_fs_type * fs);
  void               enkf_state_free(enkf_state_type * );

  const ensemble_config_type * enkf_state_get_ensemble_config( const enkf_state_type * enkf_state );

/******************************************************************/
/* Forward model callbacks: */
  bool enkf_state_complete_forward_modelOK__(void * arg );
  bool enkf_state_complete_forward_modelRETRY__(void * arg );
  bool enkf_state_complete_forward_modelEXIT__(void * arg );

  bool enkf_state_complete_forward_modelOK(const res_config_type * res_config,
                                                run_arg_type * run_arg);
  bool enkf_state_complete_forward_model_EXIT_handler__(run_arg_type * run_arg);

#ifdef __cplusplus
}
#endif
#endif
