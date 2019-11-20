/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'forward_model.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_FORWARD_MODEL_H
#define ERT_FORWARD_MODEL_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include <ert/util/stringlist.hpp>
#include <ert/res_util/subst_list.hpp>

#include <ert/job_queue/environment_varlist.hpp>
#include <ert/job_queue/ext_joblist.hpp>

typedef struct  forward_model_struct forward_model_type ;


  stringlist_type        * forward_model_alloc_joblist( const forward_model_type * forward_model );
  void                     forward_model_clear( forward_model_type * forward_model );
  void                     forward_model_fprintf(const forward_model_type *  , FILE * );
  forward_model_type     * forward_model_alloc(const ext_joblist_type * ext_joblist);
  void                     forward_model_parse_job_args(forward_model_type * model, const stringlist_type * list);
  void                     forward_model_parse_job_deprecated_args(forward_model_type * forward_model, const char * input_string); //DEPRECATED
  void                     forward_model_formatted_fprintf(const forward_model_type *  , const char * run_id, const char *, const char * , const subst_list_type * ,
                                                           mode_t umask, const env_varlist_type * list);
  void                     forward_model_free( forward_model_type * );
  void                     forward_model_iset_job_arg( forward_model_type * forward_model , int job_index , const char * arg , const char * value);
  ext_job_type           * forward_model_iget_job( forward_model_type * forward_model , int index);
  int                      forward_model_get_length( const forward_model_type * forward_model );

  ext_job_type           * forward_model_add_job(forward_model_type * forward_model , const char * job_name);


#ifdef __cplusplus
}
#endif
#endif
