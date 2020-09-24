/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'run_arg.c' is part of ERT - Ensemble based Reservoir Tool.

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


#include <ert/util/type_macros.h>

#include <ert/res_util/subst_list.hpp>

#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/run_arg.hpp>


#define RUN_ARG_TYPE_ID 66143287
#define INVALID_QUEUE_INDEX -99


struct run_arg_struct {
  UTIL_TYPE_ID_DECLARATION;
  int                     iens;
  int                     max_internal_submit;  /* How many times the enkf_state object should try to resubmit when the queueu has said everything is OK - but the load fails. */
  int                     num_internal_submit;
  int                     load_start;           /* When loading back results - start at this step. */
  int                     step1;                /* The forward model is integrated: step1 -> step2 */
  int                     step2;
  int                     iter;
  char                  * run_path;             /* The currently used  runpath - is realloced / freed for every step. */
  char                  * job_name;             /* Name of the job - will correspond to ECLBASE for eclipse jobs. */
  run_mode_type           run_mode;             /* What type of run this is */
  int                     queue_index;          /* The job will in general have a different index in the queue than the iens number. */
  int                     geo_id;               /* This will be used by WPRO - and mapped to context key <GEO_ID>; set during submit. */
  enkf_fs_type          * sim_fs;
  enkf_fs_type          * update_target_fs;
  subst_list_type       * subst_list;

  /******************************************************************/
  /* Return value - set by the called routine!!  */
  run_status_type         run_status;
  char                  * run_id;
};


UTIL_SAFE_CAST_FUNCTION( run_arg , RUN_ARG_TYPE_ID )
UTIL_IS_INSTANCE_FUNCTION( run_arg , RUN_ARG_TYPE_ID )

static void run_arg_update_subst(run_arg_type * run_arg);

static run_arg_type * run_arg_alloc(const char * run_id,
                                    enkf_fs_type * sim_fs ,
                                    enkf_fs_type * update_target_fs ,
                                    int iens ,
                                    run_mode_type run_mode          ,
                                    int step1                       ,
                                    int step2                       ,
                                    int iter                        ,
                                    const char * runpath,
				                    const char * job_name,
                                    const subst_list_type * subst_list)
{
  if ((sim_fs != NULL) && (sim_fs == update_target_fs))
    util_abort("%s: internal error - can  not have sim_fs == update_target_fs \n",__func__);
  {
    run_arg_type * run_arg = (run_arg_type *)util_malloc(sizeof * run_arg );
    UTIL_TYPE_ID_INIT(run_arg , RUN_ARG_TYPE_ID);
    run_arg->run_id = util_alloc_string_copy( run_id );
    run_arg->sim_fs = sim_fs;
    run_arg->update_target_fs = update_target_fs;

    run_arg->iens = iens;
    run_arg->run_mode = run_mode;
    run_arg->step1 = step1;
    run_arg->step2 = step2;
    run_arg->iter = iter;
    run_arg->run_path = util_alloc_abs_path( runpath );
    run_arg->job_name = util_alloc_string_copy( job_name );
    run_arg->num_internal_submit = 0;
    run_arg->queue_index = INVALID_QUEUE_INDEX;
    run_arg->run_status = JOB_NOT_STARTED;
    run_arg->geo_id = -1;    // -1 corresponds to not set
    run_arg->load_start = step1;

    run_arg->subst_list = subst_list_alloc(subst_list);
    run_arg_update_subst(run_arg);

    return run_arg;
  }
}


run_arg_type * run_arg_alloc_ENSEMBLE_EXPERIMENT(const char * run_id,
                                                 enkf_fs_type * sim_fs,
                                                 int iens,
                                                 int iter,
                                                 const char * runpath,
                                                 const char * job_name,
                                                 const subst_list_type * subst_list)
{
  return run_arg_alloc(run_id,
                       sim_fs,
                       NULL,
                       iens,
                       ENSEMBLE_EXPERIMENT,
                       0,
                       0,
                       iter,
                       runpath,
                       job_name,
                       subst_list);
}


run_arg_type * run_arg_alloc_INIT_ONLY(const char * run_id,
                                       enkf_fs_type * sim_fs,
                                       int iens,
                                       int iter,
                                       const char * runpath,
                                       const subst_list_type * subst_list)
{
  return run_arg_alloc(run_id,
                       sim_fs,
                       NULL,
                       iens,
                       INIT_ONLY,
                       0,
                       0,
                       iter,
                       runpath,
                       NULL,
                       subst_list);
}


run_arg_type * run_arg_alloc_SMOOTHER_RUN(const char * run_id,
                                          enkf_fs_type * sim_fs,
                                          enkf_fs_type * update_target_fs,
                                          int iens,
                                          int iter,
                                          const char * runpath,
                                          const char * job_name,
                                          const subst_list_type * subst_list)
{
  return run_arg_alloc(run_id,
                       sim_fs,
                       update_target_fs,
                       iens,
                       ENSEMBLE_EXPERIMENT,
                       0,
                       0,
                       iter,
                       runpath,
                       job_name,
                       subst_list);
}



void run_arg_free(run_arg_type * run_arg) {
  free( run_arg->job_name );
  free(run_arg->run_path);
  free( run_arg->run_id );
  subst_list_free(run_arg->subst_list);
  free(run_arg);
}


void run_arg_free__(void * arg) {
  run_arg_type * run_arg = run_arg_safe_cast( arg );
  run_arg_free( run_arg );
}





void run_arg_increase_submit_count( run_arg_type * run_arg ) {
  run_arg->num_internal_submit++;
}


void run_arg_set_queue_index( run_arg_type * run_arg , int queue_index) {
  if (run_arg->queue_index == INVALID_QUEUE_INDEX)
    run_arg->queue_index = queue_index;
  else
    util_abort("%s: attempt to reset run_arg->queue_index. These objects should not be recycled\n",__func__);
}



const char * run_arg_get_runpath( const run_arg_type * run_arg) {
  return run_arg->run_path;
}


const char * run_arg_get_job_name( const run_arg_type * run_arg) {
  return run_arg->job_name;
}


const char * run_arg_get_run_id( const run_arg_type * run_arg) {
  return run_arg->run_id;
}




int run_arg_get_iter( const run_arg_type * run_arg ) {
  return run_arg->iter;
}


int run_arg_get_iens( const run_arg_type * run_arg ) {
  return run_arg->iens;
}


int run_arg_get_load_start( const run_arg_type * run_arg ) {
  return run_arg->load_start;
}


int run_arg_get_step2( const run_arg_type * run_arg ) {
  return run_arg->step2;
}

bool run_arg_can_retry( const run_arg_type * run_arg ) {
  if (run_arg->num_internal_submit < run_arg->max_internal_submit)
    return true;
  else
    return false;
}


int run_arg_get_step1( const run_arg_type * run_arg ) {
  return run_arg->step1;
}


int run_arg_get_queue_index_safe( const run_arg_type * run_arg ) {
  if (run_arg->queue_index == INVALID_QUEUE_INDEX)
    return -1;

  return run_arg->queue_index;
}

int run_arg_get_queue_index( const run_arg_type * run_arg ) {
  if (run_arg->queue_index == INVALID_QUEUE_INDEX)
    util_abort("%s: sorry internal error - asking for the queue_index in a not-initialized run_arg object.\n" , __func__);

  return run_arg->queue_index;
}

bool run_arg_is_submitted( const run_arg_type * run_arg ) {
  if (run_arg->queue_index == INVALID_QUEUE_INDEX)
    return false;
  else
    return true;
}


run_status_type run_arg_get_run_status( const run_arg_type * run_arg) {
  return run_arg->run_status;
}


void run_arg_set_run_status( run_arg_type * run_arg , run_status_type run_status) {
  run_arg->run_status = run_status;
}


void run_arg_set_geo_id( run_arg_type * run_arg , int geo_id) {
  /*
   * Providing the geo_id upon initialization is the last step to achieve
   * immutability for run_arg. Although we allow setting the geo_id for now, it
   * should only be done once.
   */
  if(run_arg->geo_id != -1)
    util_abort("%s: Tried to set run_arg's geo_id twice!\n", __func__);

  run_arg->geo_id = geo_id;
  run_arg_update_subst(run_arg);
}


int run_arg_get_geo_id( const run_arg_type * run_arg) {
  return run_arg->geo_id;
}


enkf_fs_type * run_arg_get_sim_fs(const run_arg_type * run_arg) {
  if (run_arg->sim_fs)
    return run_arg->sim_fs;
  else {
    util_abort("%s: internal error - tried to access run_arg->sim_fs when sim_fs == NULL\n",__func__);
    return NULL;
  }
}


enkf_fs_type * run_arg_get_update_target_fs(const run_arg_type * run_arg) {
  if (run_arg->update_target_fs)
    return run_arg->update_target_fs;
  else {
    util_abort("%s: internal error - tried to access run_arg->update_target_fs when update_target_fs == NULL\n",__func__);
    return NULL;
  }
}

const subst_list_type * run_arg_get_subst_list(const run_arg_type * run_arg)
{
  return run_arg->subst_list;
}

static void run_arg_update_subst(run_arg_type * run_arg)
{
  char * iens_str = util_alloc_sprintf("%d", run_arg->iens);
  subst_list_prepend_owned_ref(run_arg->subst_list, "<IENS>", iens_str, NULL);

  char * iter_str = util_alloc_sprintf("%d", run_arg->iter);
  subst_list_prepend_owned_ref(run_arg->subst_list, "<ITER>", iter_str, NULL);

  if (run_arg->geo_id != -1) {
    char * geo_id_str = util_alloc_sprintf("%d", run_arg->geo_id);
    subst_list_prepend_owned_ref(run_arg->subst_list, "<GEO_ID>", geo_id_str, NULL);
  }

  if(run_arg->job_name) {
    subst_list_update_string(run_arg->subst_list, &run_arg->job_name);
    subst_list_prepend_ref(run_arg->subst_list, "<ECL_BASE>", run_arg->job_name, NULL);
    subst_list_prepend_ref(run_arg->subst_list, "<ECLBASE>", run_arg->job_name, NULL);
  }

  subst_list_update_string(run_arg->subst_list, &run_arg->run_path);

  subst_list_prepend_ref(run_arg->subst_list, "<RUNPATH>", run_arg->run_path, NULL);
}
