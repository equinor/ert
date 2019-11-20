/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'ert_run_context.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <unistd.h>
#include <sys/types.h>
#include <time.h>

#include <ert/util/type_macros.h>
#include <ert/util/vector.h>
#include <ert/util/int_vector.h>
#include <ert/util/stringlist.h>
#include <ert/util/type_vector_functions.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/res_util/path_fmt.hpp>

#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/run_arg.hpp>
#include <ert/enkf/ert_run_context.hpp>


#define ERT_RUN_CONTEXT_TYPE_ID 55534132


struct ert_run_context_struct {
  UTIL_TYPE_ID_DECLARATION;
  vector_type      * run_args;
  run_mode_type      run_mode;
  init_mode_type     init_mode;
  int                iter;
  int                step1;
  int                step2;
  int                load_start;
  int_vector_type  * iens_map;
  bool_vector_type * iactive;
  enkf_fs_type          * sim_fs;
  enkf_fs_type          * update_target_fs;
  char                  * run_id;
};



/*
  Observe that since this function uses one shared subst_list instance
  for all realisations it is __NOT__ possible to get per realisation
  substititions in the runpath; i.e. a <IENS> element in the RUNPATH
  will *not* be replaced.
*/


char * ert_run_context_alloc_runpath( int iens , const path_fmt_type * runpath_fmt , const subst_list_type * subst_list , int iter) {
  char * runpath;
  {
    char * first_pass = path_fmt_alloc_path(runpath_fmt , false , iens, iter);    /* 1: Replace first %d with iens, if a second %d replace with iter */

    if (subst_list)
      runpath = subst_list_alloc_filtered_string( subst_list , first_pass );         /* 2: Filter out various magic strings like <CASE> and <CWD>. */
    else
      runpath = util_alloc_string_copy( first_pass );

    free( first_pass );
  }
  return runpath;
}

static char * ert_run_context_alloc_jobname( int iens , const char * jobname_fmt , const subst_list_type * subst_list) {
  char * jobname;
  {
    char * first_pass = util_alloc_sprintf( jobname_fmt, iens );

    if (subst_list)
      jobname = subst_list_alloc_filtered_string( subst_list , first_pass );         /* 2: Filter out various magic strings like <CASE> and <CWD>. */
    else
      jobname = util_alloc_string_copy( first_pass );

    free( first_pass );
  }
  return jobname;
}



stringlist_type * ert_run_context_alloc_runpath_list(const bool_vector_type * iactive , const path_fmt_type * runpath_fmt , const subst_list_type * subst_list , int iter) {
  stringlist_type * runpath_list = stringlist_alloc_new();
  for (int iens = 0; iens < bool_vector_size( iactive ); iens++) {

    if (bool_vector_iget( iactive , iens )) {
      char * tmp_runpath = ert_run_context_alloc_runpath(iens, runpath_fmt, subst_list, iter);
      stringlist_append_copy( runpath_list , tmp_runpath);
      free(tmp_runpath);
    } else
      stringlist_append_copy( runpath_list , NULL );

  }
  return runpath_list;
}


static stringlist_type * ert_run_context_alloc_jobname_list(const bool_vector_type * iactive , const char * jobname_fmt, const subst_list_type * subst_list) {
  stringlist_type * jobname_list = stringlist_alloc_new();
  for (int iens = 0; iens < bool_vector_size( iactive ); iens++) {

    if (bool_vector_iget( iactive , iens )) {
      char * tmp_jobname = ert_run_context_alloc_jobname(iens, jobname_fmt, subst_list);
      stringlist_append_copy( jobname_list, tmp_jobname);
      free(tmp_jobname);
    } else
      stringlist_append_copy( jobname_list , NULL );

  }
  return jobname_list;
}



char * ert_run_context_alloc_run_id( ) {
  int year,month,day,hour,min,sec;
  time_t now = time( NULL );
  unsigned int random = util_dev_urandom_seed( );
  util_set_datetime_values_utc( now , &sec, &min, &hour, &day, &month, &year);
  return util_alloc_sprintf("%d:%d:%4d-%0d-%02d-%02d-%02d-%02d:%ud" , getpid() , getuid(), year , month , day , hour , min , sec, random);
}

static ert_run_context_type * ert_run_context_alloc__(const bool_vector_type * iactive , run_mode_type run_mode , init_mode_type init_mode, enkf_fs_type * sim_fs , enkf_fs_type * update_target_fs , int iter) {
  ert_run_context_type * context = (ert_run_context_type *)util_malloc( sizeof * context );
  UTIL_TYPE_ID_INIT( context , ERT_RUN_CONTEXT_TYPE_ID );

  if (iactive != NULL) {
    context->iactive = bool_vector_alloc_copy( iactive );
    context->iens_map = bool_vector_alloc_active_index_list( iactive , -1 );
  }
  else {
    context->iactive = NULL;
    context->iens_map = NULL;
  }
  context->run_args = vector_alloc_new();
  context->run_mode = run_mode;
  context->init_mode = init_mode;
  context->iter = iter;
  ert_run_context_set_sim_fs(context, sim_fs);
  ert_run_context_set_update_target_fs(context, update_target_fs);

  context->step1 = 0;
  context->step2 = 0;
  context->run_id = ert_run_context_alloc_run_id( );
  return context;
}


ert_run_context_type * ert_run_context_alloc_ENSEMBLE_EXPERIMENT(enkf_fs_type * sim_fs,
                                                                 bool_vector_type * iactive ,
                                                                 const path_fmt_type * runpath_fmt ,
                                                                 const char * jobname_fmt,
                                                                 const subst_list_type * subst_list ,
                                                                 int iter) {

  ert_run_context_type * context = ert_run_context_alloc__( iactive , ENSEMBLE_EXPERIMENT , INIT_CONDITIONAL, sim_fs , NULL , iter);
  {
    stringlist_type * runpath_list = ert_run_context_alloc_runpath_list( iactive , runpath_fmt , subst_list , iter );
    stringlist_type * jobname_list = ert_run_context_alloc_jobname_list( iactive , jobname_fmt , subst_list );
    for (int iens = 0; iens < bool_vector_size( iactive ); iens++) {
      if (bool_vector_iget( iactive , iens )) {
        run_arg_type * arg = run_arg_alloc_ENSEMBLE_EXPERIMENT( context->run_id,
                                                                sim_fs,
                                                                iens,
                                                                iter,
                                                                stringlist_iget( runpath_list , iens),
                                                                stringlist_iget( jobname_list, iens),
                                                                subst_list);
        vector_append_owned_ref( context->run_args , arg , run_arg_free__);
      } else
        vector_append_ref( context->run_args, NULL );
    }
    stringlist_free( jobname_list );
    stringlist_free( runpath_list );
  }
  return context;
}


ert_run_context_type * ert_run_context_alloc_INIT_ONLY(enkf_fs_type * sim_fs,
                                                       init_mode_type init_mode,
                                                       bool_vector_type * iactive ,
                                                       const path_fmt_type * runpath_fmt ,
                                                       const subst_list_type * subst_list ,
                                                       int iter) {
  ert_run_context_type * context = ert_run_context_alloc__( iactive , INIT_ONLY , init_mode, sim_fs, NULL , iter);
  {
    stringlist_type * runpath_list = ert_run_context_alloc_runpath_list( iactive , runpath_fmt , subst_list , iter );
    for (int iens = 0; iens < bool_vector_size( iactive ); iens++) {
      if (bool_vector_iget( iactive , iens )) {
        run_arg_type * arg = run_arg_alloc_INIT_ONLY(context->run_id,
                                                     sim_fs,
                                                     iens,
                                                     iter,
                                                     stringlist_iget(runpath_list, iens),
                                                     subst_list);
        vector_append_owned_ref( context->run_args , arg , run_arg_free__);
      } else
        vector_append_ref( context->run_args, NULL );
    }
    stringlist_free( runpath_list );
  }
  return context;
}


ert_run_context_type * ert_run_context_alloc_CASE_INIT(enkf_fs_type * sim_fs,
                                                       const bool_vector_type * iactive) {
  ert_run_context_type * run_context = ert_run_context_alloc__(iactive, CASE_INIT_ONLY, INIT_FORCE, sim_fs, NULL, 0);
  return run_context;
}


ert_run_context_type * ert_run_context_alloc_SMOOTHER_RUN(enkf_fs_type * sim_fs , enkf_fs_type * target_update_fs ,
                                                          bool_vector_type * iactive ,
                                                          const path_fmt_type * runpath_fmt ,
							  const char * jobname_fmt,
							  const subst_list_type * subst_list ,
                                                          int iter) {

  ert_run_context_type * context = ert_run_context_alloc__( iactive , SMOOTHER_RUN , INIT_CONDITIONAL, sim_fs , target_update_fs , iter);
  {
    stringlist_type * runpath_list = ert_run_context_alloc_runpath_list( iactive , runpath_fmt , subst_list , iter );
    stringlist_type * jobname_list = ert_run_context_alloc_jobname_list( iactive , jobname_fmt , subst_list );
    for (int iens = 0; iens < bool_vector_size( iactive ); iens++) {
      if (bool_vector_iget( iactive , iens )) {
        run_arg_type * arg = run_arg_alloc_SMOOTHER_RUN( context->run_id,
							 sim_fs,
							 target_update_fs,
							 iens,
							 iter,
							 stringlist_iget( runpath_list , iens),
							 stringlist_iget( jobname_list , iens),
                             subst_list);
        vector_append_owned_ref( context->run_args , arg , run_arg_free__);
      } else
        vector_append_ref( context->run_args, NULL );
    }
    stringlist_free( jobname_list );
    stringlist_free( runpath_list );
  }
  return context;
}

ert_run_context_type * ert_run_context_alloc_SMOOTHER_UPDATE(enkf_fs_type * sim_fs , enkf_fs_type * target_update_fs ) {
  return ert_run_context_alloc__( NULL , SMOOTHER_UPDATE , INIT_CONDITIONAL, sim_fs , target_update_fs , 0);
}

ert_run_context_type * ert_run_context_alloc(run_mode_type run_mode,
                                             init_mode_type init_mode,
                                             enkf_fs_type * sim_fs ,
                                             enkf_fs_type * target_update_fs ,
                                             bool_vector_type * iactive ,
                                             path_fmt_type * runpath_fmt ,
                                             const char * jobname_fmt,
                                             subst_list_type * subst_list ,
                                             int iter) {
  switch (run_mode) {

  case(SMOOTHER_RUN):
    return ert_run_context_alloc_SMOOTHER_RUN( sim_fs , target_update_fs, iactive, runpath_fmt , jobname_fmt, subst_list , iter );

  case(ENSEMBLE_EXPERIMENT):
    return ert_run_context_alloc_ENSEMBLE_EXPERIMENT( sim_fs , iactive , runpath_fmt , jobname_fmt, subst_list , iter);

  case(INIT_ONLY):
    return ert_run_context_alloc_INIT_ONLY( sim_fs, init_mode , iactive, runpath_fmt , subst_list, iter );

  case(SMOOTHER_UPDATE):
    break;

  case(CASE_INIT_ONLY):
    break;

  }

  util_abort("%s: internal error - should never be here \n",__func__);
  return NULL;
}





UTIL_IS_INSTANCE_FUNCTION( ert_run_context , ERT_RUN_CONTEXT_TYPE_ID );


const char * ert_run_context_get_id( const ert_run_context_type * context ) {
  return context->run_id;
}


void ert_run_context_free( ert_run_context_type * context ) {
  if (context->sim_fs) {
    enkf_fs_decrease_run_count(context->sim_fs);
    enkf_fs_decref(context->sim_fs);
  }

  if (context->update_target_fs) {
    enkf_fs_decrease_run_count(context->update_target_fs);
    enkf_fs_decref(context->update_target_fs);
  }

  vector_free( context->run_args );
  if (context->iactive != NULL) {
    int_vector_free( context->iens_map );
    bool_vector_free( context->iactive );
  }
  free( context->run_id );
  free( context );
}


int ert_run_context_get_size( const ert_run_context_type * context ) {
  return vector_get_size( context->run_args );
}



run_mode_type ert_run_context_get_mode( const ert_run_context_type * context ) {
  return context->run_mode;
}





int ert_run_context_get_iter( const ert_run_context_type * context ) {
  return context->iter;
}


init_mode_type ert_run_context_get_init_mode( const ert_run_context_type * context ) {
  return context->init_mode;
}

int ert_run_context_get_step1( const ert_run_context_type * context ) {
  return context->step1;
}

run_arg_type * ert_run_context_iget_arg( const ert_run_context_type * context , int index) {
  return (run_arg_type * ) vector_iget( context->run_args , index );
}


run_arg_type * ert_run_context_iens_get_arg( const ert_run_context_type * context , int iens) {
  int index = int_vector_iget( context->iens_map , iens );
  if (index >= 0)
    return (run_arg_type * ) vector_iget( context->run_args , index );
  else
    return NULL;
}

enkf_fs_type * ert_run_context_get_sim_fs(const ert_run_context_type * run_context) {
  if (run_context->sim_fs)
    return run_context->sim_fs;
  else {
    util_abort("%s: internal error - tried to access run_context->sim_fs when sim_fs == NULL\n",__func__);
    return NULL;
  }
}


enkf_fs_type * ert_run_context_get_update_target_fs(const ert_run_context_type * run_context) {
  if (run_context->update_target_fs)
    return run_context->update_target_fs;
  else {
    util_abort("%s: internal error - tried to access run_context->update_target_fs when update_target_fs == NULL\n",__func__);
    return NULL;
  }
}

void ert_run_context_set_sim_fs(ert_run_context_type * context, enkf_fs_type * sim_fs) {
  if (sim_fs) {
    context->sim_fs = sim_fs;
    enkf_fs_increase_run_count(sim_fs);
    enkf_fs_incref(sim_fs);
  } else
    context->sim_fs = NULL;
}

void ert_run_context_set_update_target_fs(ert_run_context_type * context, enkf_fs_type * update_target_fs) {
  if (update_target_fs) {
    context->update_target_fs = update_target_fs;
    enkf_fs_increase_run_count(update_target_fs);
    enkf_fs_incref(update_target_fs);
  } else
    context->update_target_fs = NULL;
}



void ert_run_context_deactivate_realization( ert_run_context_type * context , int iens) {
  bool_vector_iset( context->iactive , iens , false );
}


bool ert_run_context_iactive( const ert_run_context_type * context , int iens) {
  return bool_vector_safe_iget( context->iactive , iens );
}

int ert_run_context_get_active_size(const ert_run_context_type * context){
  return bool_vector_count_equal(context->iactive, true);
}

bool_vector_type * ert_run_context_alloc_iactive(const ert_run_context_type * context) {
  return bool_vector_alloc_copy(context->iactive);
}

bool_vector_type const * ert_run_context_get_iactive(const ert_run_context_type * context) {
   return context->iactive;
}
