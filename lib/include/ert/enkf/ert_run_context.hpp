/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'ert_run_context.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_RUN_CONTEXT_H
#define ERT_RUN_CONTEXT_H

#include <ert/util/type_macros.h>
#include <ert/util/bool_vector.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/res_util/path_fmt.hpp>

#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/run_arg.hpp>
#include <ert/enkf/enkf_fs.hpp>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ert_run_context_struct ert_run_context_type;

  stringlist_type      * ert_run_context_alloc_runpath_list(const bool_vector_type * iactive ,
                                                            const path_fmt_type * runpath_fmt ,
                                                            const subst_list_type * subst_list ,
                                                            int iter);

  char                 * ert_run_context_alloc_runpath( int iens , const path_fmt_type * runpath_fmt , const subst_list_type * subst_list , int iter);

  ert_run_context_type * ert_run_context_alloc(run_mode_type run_mode,
                                               init_mode_type init_mode,
                                               enkf_fs_type * sim_fs ,
                                               enkf_fs_type * target_update_fs ,
                                               bool_vector_type * iactive ,
                                               path_fmt_type * runpath_fmt ,
                                               const char * jobname_fmt,
                                               subst_list_type * subst_list ,
                                               int iter);

  ert_run_context_type * ert_run_context_alloc_ENSEMBLE_EXPERIMENT(enkf_fs_type * sim_fs ,
                                                                   bool_vector_type * iactive ,
                                                                   const path_fmt_type * runpath_fmt ,
                                                                   const char * jobname_fmt,
                                                                   const subst_list_type * subst_list ,
                                                                   int iter);

  ert_run_context_type * ert_run_context_alloc_INIT_ONLY(enkf_fs_type * sim_fs,
                                                         init_mode_type init_mode,
                                                         bool_vector_type * iactive ,
                                                         const path_fmt_type * runpath_fmt ,
                                                         const subst_list_type * subst_list ,
                                                         int iter);

  ert_run_context_type * ert_run_context_alloc_SMOOTHER_RUN(enkf_fs_type * sim_fs , enkf_fs_type * target_update_fs ,
                                                            bool_vector_type * iactive ,
                                                            const path_fmt_type * runpath_fmt ,
                                                            const char * jobname_fmt,
                                                            const subst_list_type * subst_list ,
                                                            int iter);

  ert_run_context_type * ert_run_context_alloc_SMOOTHER_UPDATE(enkf_fs_type * sim_fs , enkf_fs_type * target_update_fs );

  ert_run_context_type * ert_run_context_alloc_CASE_INIT(enkf_fs_type * sim_fs,
                                                         const bool_vector_type * iactive);

  void                     ert_run_context_set_sim_fs(ert_run_context_type * context, enkf_fs_type * sim_fs);
  void                     ert_run_context_set_update_target_fs(ert_run_context_type * context, enkf_fs_type * update_target_fs);

  void                     ert_run_context_free( ert_run_context_type * );
  int                      ert_run_context_get_size( const ert_run_context_type * context );
  run_mode_type            ert_run_context_get_mode( const ert_run_context_type * context );
  bool_vector_type       * ert_run_context_alloc_iactive(const ert_run_context_type * context);
  bool_vector_type const * ert_run_context_get_iactive(const ert_run_context_type * context);
  int                      ert_run_context_get_iter( const ert_run_context_type * context );
  int                      ert_run_context_get_active_size(const ert_run_context_type * context);
  int                      ert_run_context_get_step1( const ert_run_context_type * context );
  run_arg_type           * ert_run_context_iget_arg( const ert_run_context_type * context , int index);
  run_arg_type           * ert_run_context_iens_get_arg( const ert_run_context_type * context , int iens);
  void                     ert_run_context_deactivate_realization( ert_run_context_type * context , int iens);
  const char             * ert_run_context_get_id( const ert_run_context_type * context );
  init_mode_type           ert_run_context_get_init_mode( const ert_run_context_type * context );
  char                   * ert_run_context_alloc_run_id( );

  enkf_fs_type * ert_run_context_get_sim_fs(const ert_run_context_type * run_context);
  enkf_fs_type * ert_run_context_get_update_target_fs(const ert_run_context_type * run_context);
  bool ert_run_context_iactive( const ert_run_context_type * context , int iens);

  UTIL_IS_INSTANCE_HEADER( ert_run_context );


#ifdef __cplusplus
}
#endif
#endif
