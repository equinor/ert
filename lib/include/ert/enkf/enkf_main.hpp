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

#include <stdbool.h>

#include <ert/util/util.h>
#include <ert/util/stringlist.h>
#include <ert/util/bool_vector.h>
#include <ert/util/int_vector.h>
#include <ert/res_util/matrix.hpp>
#include <ert/res_util/ui_return.hpp>
#include <ert/res_util/log.hpp>
#include <ert/res_util/subst_list.hpp>

#include <ert/config/config_settings.hpp>

#include <ert/job_queue/job_queue.hpp>
#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/forward_model.hpp>

#include <ert/enkf/res_config.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/misfit_ensemble.hpp>
#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/site_config.hpp>
#include <ert/enkf/local_config.hpp>
#include <ert/enkf/ert_template.hpp>
#include <ert/enkf/enkf_plot_data.hpp>
#include <ert/enkf/ranking_table.hpp>
#include <ert/enkf/hook_manager.hpp>
#include <ert/enkf/rng_config.hpp>
#include <ert/enkf/rng_manager.hpp>
#include <ert/enkf/field_config.hpp>
#include <ert/enkf/ert_run_context.hpp>

#ifdef __cplusplus
extern "C" {
#endif

  /*****************************************************************/


  typedef struct enkf_main_struct enkf_main_type;
  const char                  * enkf_main_get_user_config_file( const enkf_main_type * enkf_main );

  ert_templates_type          * enkf_main_get_templates( enkf_main_type * enkf_main );

  void                          enkf_main_free(enkf_main_type * );
  void                          enkf_main_exit(enkf_main_type * enkf_main);
  void                          enkf_main_init_run( enkf_main_type * enkf_main, const ert_run_context_type * run_context);

  bool                          enkf_main_UPDATE(enkf_main_type * enkf_main , const int_vector_type * step_list, enkf_fs_type * source_fs, enkf_fs_type * target_fs , int target_step , run_mode_type run_mode);
  bool                          enkf_main_smoother_update(enkf_main_type * enkf_main , enkf_fs_type * source_fs, enkf_fs_type * target_fs);
  void                          enkf_main_create_run_path(enkf_main_type * enkf_main , const ert_run_context_type * run_context);

  enkf_main_type              * enkf_main_alloc(const res_config_type *, bool, bool);

  enkf_state_type             * enkf_main_iget_state(const enkf_main_type * , int );

  ranking_table_type          * enkf_main_get_ranking_table( const enkf_main_type * enkf_main );
  const ecl_config_type       * enkf_main_get_ecl_config(const enkf_main_type * enkf_main);
  ensemble_config_type        * enkf_main_get_ensemble_config(const enkf_main_type * enkf_main);
  int                           enkf_main_get_ensemble_size( const enkf_main_type * enkf_main );
  int                           enkf_main_get_history_length( const enkf_main_type * );
  model_config_type           * enkf_main_get_model_config( const enkf_main_type * );
  local_config_type           * enkf_main_get_local_config( const enkf_main_type * enkf_main );
  bool                          enkf_main_load_obs(enkf_main_type *, const char *, bool);
  enkf_obs_type               * enkf_main_get_obs(const enkf_main_type * );
  bool                          enkf_main_have_obs( const enkf_main_type * enkf_main );
  const analysis_config_type  * enkf_main_get_analysis_config(const enkf_main_type * );
  const log_config_type       * enkf_main_get_log_config(const enkf_main_type *);

  subst_config_type      * enkf_main_get_subst_config(const enkf_main_type * enkf_main);
  subst_list_type        * enkf_main_get_data_kw( const enkf_main_type * enkf_main );
  PY_USED void             enkf_main_clear_data_kw( enkf_main_type * enkf_main );
  const site_config_type * enkf_main_get_site_config( const enkf_main_type * enkf_main );
  void                     enkf_main_resize_ensemble( enkf_main_type * enkf_main , int new_ens_size );
  void                     enkf_main_get_observations( const enkf_main_type * enkf_main, const char * user_key , int obs_count , time_t * obs_time , double * y , double * std);
  int                      enkf_main_get_observation_count( const enkf_main_type * enkf_main, const char * user_key );

  /*****************************************************************/
  void                        enkf_main_install_SIGNALS(void);
  void                        enkf_main_add_node(enkf_main_type * enkf_main, enkf_config_node_type * enkf_config_node);
  int_vector_type           * enkf_main_update_alloc_step_list( const enkf_main_type * enkf_main , int load_start , int step2 , int stride);

  const hook_manager_type   * enkf_main_get_hook_manager( const enkf_main_type * enkf_main );

  void enkf_main_get_PC( const matrix_type * S,
                         const matrix_type * dObs,
                         double truncation ,
                         int ncomp ,
                         matrix_type * PC ,
                         matrix_type * PC_obs,
                         double_vector_type * singular_values);


  void enkf_main_fprintf_PC(const char * filename ,
                            matrix_type * PC ,
                            matrix_type * PC_obs);


  void                   enkf_main_set_verbose( enkf_main_type * enkf_main , bool verbose);

  ert_workflow_list_type * enkf_main_get_workflow_list( enkf_main_type * enkf_main );
  bool                     enkf_main_run_workflow( enkf_main_type * enkf_main , const char * workflow);

  rng_config_type     * enkf_main_get_rng_config( const enkf_main_type * enkf_main );
  void                  enkf_main_rng_init( enkf_main_type * enkf_main);
  rng_type            * enkf_main_get_shared_rng( enkf_main_type * enkf_main );

  bool enkf_main_export_field(const enkf_main_type * enkf_main,
                              const char * kw,
                              const char * path,
                              bool_vector_type * iactive,
                              field_file_format_type file_type,
                              int report_step);

  bool enkf_main_export_field_with_fs(const enkf_main_type * enkf_main,
                              const char * kw,
                              const char * path,
                              bool_vector_type * iactive,
                              field_file_format_type file_type,
                              int report_step,
                              enkf_fs_type * fs);


  int enkf_main_load_from_forward_model_with_fs(enkf_main_type * enkf_main, int iter , bool_vector_type * iactive, stringlist_type ** realizations_msg_list, enkf_fs_type * fs);
  int enkf_main_load_from_forward_model(enkf_main_type * enkf_main, int iter , bool_vector_type * iactive, stringlist_type ** realizations_msg_list);
  PY_USED int enkf_main_load_from_forward_model_from_gui(enkf_main_type * enkf_main, int iter , bool_vector_type * iactive, enkf_fs_type * fs);
  int enkf_main_load_from_run_context(enkf_main_type* enkf_main, ert_run_context_type* run_context, stringlist_type** realizations_msg_list, enkf_fs_type* fs);
  int enkf_main_load_from_run_context_from_gui(enkf_main_type* enkf_main, ert_run_context_type* run_context, enkf_fs_type* fs);

  void enkf_main_rank_on_observations(enkf_main_type * enkf_main,
                                      const char * ranking_key,
                                      const stringlist_type * obs_ranking_keys,
                                      const int_vector_type * steps);



  void enkf_main_rank_on_data(enkf_main_type * enkf_main,
                              const char * ranking_key,
                              const char * data_key,
                              bool sort_increasing,
                              int step);


  void enkf_main_export_ranking(enkf_main_type * enkf_main, const char * ranking_key, const char * ranking_file);







  /********* File System / Case Management **********
          Implementation: enkf_main_manage_fs.c     */


  bool              enkf_main_case_is_current(const enkf_main_type * enkf_main , const char * case_path);
  char *            enkf_main_read_alloc_current_case_name(const enkf_main_type * enkf_main);
  stringlist_type * enkf_main_alloc_caselist( const enkf_main_type * enkf_main );

  void              enkf_main_initialize_from_scratch(enkf_main_type * enkf_main ,
                                                      const stringlist_type * param_list ,
                                                      const ert_run_context_type * run_context);

  void              enkf_main_init_current_case_from_existing(enkf_main_type * enkf_main,
                                                              enkf_fs_type * source_case_fs,
                                                              int source_report_step);

  PY_USED void      enkf_main_init_current_case_from_existing_custom(enkf_main_type * enkf_main,
                                                                    enkf_fs_type * source_case_fs,
                                                                    int source_report_step,
                                                                    stringlist_type * node_list,
                                                                    bool_vector_type * iactive);


  void              enkf_main_init_case_from_existing(const enkf_main_type * enkf_main,
                                                      enkf_fs_type * source_case_fs,
                                                      int source_report_step,
                                                      enkf_fs_type * target_case);

  void              enkf_main_init_case_from_existing_custom(const enkf_main_type * enkf_main,
                                                             enkf_fs_type * source_case_fs,
                                                             int source_report_step,
                                                             enkf_fs_type * target_case,
                                                             stringlist_type * node_list,
                                                             bool_vector_type * iactive);

  bool              enkf_main_case_is_initialized( const enkf_main_type * enkf_main ,
                                                   const char * case_name ,
                                                   bool_vector_type * __mask);

  bool              enkf_main_is_initialized( const enkf_main_type * enkf_main ,bool_vector_type * __mask);

  char            * enkf_main_alloc_mount_point( const enkf_main_type * enkf_main , const char * case_path);
  enkf_fs_type    * enkf_main_get_fs(const enkf_main_type * );
  enkf_fs_type    * enkf_main_job_get_fs(const enkf_main_type * );
  enkf_fs_type    * enkf_main_get_fs_ref(const enkf_main_type * enkf_main);
  const char      * enkf_main_get_current_fs( const enkf_main_type * enkf_main );
  enkf_fs_type    * enkf_main_mount_alt_fs(const enkf_main_type * enkf_main , const char * case_path ,  bool create);
  void              enkf_main_set_fs( enkf_main_type * enkf_main , enkf_fs_type * fs , const char * case_path );
  void              enkf_main_select_fs( enkf_main_type * enkf_main , const char * case_path );
  bool              enkf_main_fs_exists(const enkf_main_type * enkf_main, const char * input_case);
  const      char * enkf_main_get_mount_root( const enkf_main_type * enkf_main);


  state_map_type  * enkf_main_alloc_readonly_state_map( const enkf_main_type * enkf_main , const char * case_path);
  PY_USED time_map_type * enkf_main_alloc_readonly_time_map( const enkf_main_type * enkf_main , const char * case_path );

  runpath_list_type    * enkf_main_alloc_runpath_list(const enkf_main_type * enkf_main);
  runpath_list_type    * enkf_main_get_runpath_list( const enkf_main_type * enkf_main );
  PY_USED ert_run_context_type * enkf_main_alloc_ert_run_context_ENSEMBLE_EXPERIMENT(const enkf_main_type * enkf_main , enkf_fs_type * fs , bool_vector_type * iactive , int iter);

  queue_config_type * enkf_main_get_queue_config(enkf_main_type * enkf_main );

  rng_manager_type  * enkf_main_get_rng_manager(const enkf_main_type * enkf_main );
  void enkf_main_isubmit_job( enkf_main_type * enkf_main , run_arg_type * run_arg , job_queue_type * job_queue);
  const char * enkf_main_get_site_config_file( const enkf_main_type * enkf_main );
  const char * enkf_main_get_schedule_prediction_file( const enkf_main_type * enkf_main );
  void enkf_main_add_data_kw(enkf_main_type * enkf_main , const char * key , const char * value);
  const res_config_type * enkf_main_get_res_config(const enkf_main_type * enkf_main);
  void * enkf_main_isubmit_job__( void * arg );

UTIL_SAFE_CAST_HEADER(enkf_main);
UTIL_IS_INSTANCE_HEADER(enkf_main);

#ifdef __cplusplus
}
#endif
#endif
