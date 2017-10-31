/*
   Copyright (C) 2011  Statoil ASA, Norway.

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
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include <ert/util/hash.h>
#include <ert/util/rng.h>
#include <ert/util/stringlist.h>
#include <ert/util/matrix.h>
#include <ert/util/rng.h>
#include <ert/res_util/subst_list.h>

#include <ert/sched/sched_file.h>

#include <ert/ecl/fortio.h>
#include <ert/ecl/ecl_file.h>

#include <ert/job_queue/forward_model.h>
#include <ert/job_queue/ext_joblist.h>
#include <ert/job_queue/job_queue.h>

#include <ert/enkf/model_config.h>
#include <ert/enkf/site_config.h>
#include <ert/enkf/ecl_config.h>
#include <ert/enkf/ensemble_config.h>
#include <ert/enkf/res_config.h>
#include <ert/enkf/ert_template.h>
#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/enkf_types.h>
#include <ert/enkf/enkf_node.h>
#include <ert/enkf/enkf_util.h>
#include <ert/enkf/enkf_serialize.h>
#include <ert/enkf/run_arg.h>

typedef struct enkf_state_struct    enkf_state_type;

  //void             * enkf_state_complete_forward_model__(void * arg );
  void *             enkf_state_load_from_forward_model_mt( void * arg );
  void               enkf_state_initialize(enkf_state_type * enkf_state , rng_type * rng, enkf_fs_type * fs, const stringlist_type * param_list , init_mode_type init_mode);
  void               enkf_state_swapout_node(const enkf_state_type * , const char *);
  void               enkf_state_swapin_node(const enkf_state_type *  , const char *);
  void               enkf_state_iset_eclpath(enkf_state_type * , int , const char *);
  //enkf_node_type   * enkf_state_get_node(const enkf_state_type * , const char * );
  void               enkf_state_load_ecl_summary(enkf_state_type * , bool , int );
  void             * enkf_state_run_eclipse__(void * );
  void             * enkf_state_start_forward_model__(void * );

  int                enkf_state_load_from_forward_model(enkf_state_type * enkf_state ,
                                                        run_arg_type * run_arg ,
                                                        stringlist_type * msg_list);

  int enkf_state_forward_init(const ensemble_config_type * ens_config,
			      run_arg_type * run_arg);

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
  void               enkf_state_load_ecl_restart(enkf_state_type * , bool , int );
  void               enkf_state_sample(enkf_state_type * , int);
  void               enkf_state_ens_read(       enkf_state_type * , const char * , int);
  void               enkf_state_ecl_write(const ensemble_config_type * ens_config, const model_config_type * model_config, const run_arg_type * run_arg , enkf_fs_type * fs);
  void               enkf_state_free(enkf_state_type * );
  void               enkf_state_apply(enkf_state_type * , enkf_node_ftype1 * , int );
  void               enkf_state_serialize(enkf_state_type * , size_t);
  void               enkf_state_set_iens(enkf_state_type *  , int );
  int                enkf_state_get_iens(const enkf_state_type * );
  const char       * enkf_state_get_run_path(const enkf_state_type * );

  run_status_type    enkf_state_get_simple_run_status(const enkf_state_type * state);
  const ensemble_config_type * enkf_state_get_ensemble_config( const enkf_state_type * enkf_state );

/******************************************************************/
/* Forward model callbacks: */
  bool enkf_state_complete_forward_modelOK__(void * arg );
  bool enkf_state_complete_forward_modelRETRY__(void * arg );
  bool enkf_state_complete_forward_modelEXIT__(void * arg );


#ifdef __cplusplus
}
#endif
#endif
