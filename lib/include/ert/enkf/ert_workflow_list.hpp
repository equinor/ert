/*
   Copyright (C) 2012  Equinor ASA, Norway.

   The file 'ert_workflow_list.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_WORKFLOW_LIST_H
#define ERT_WORKFLOW_LIST_H


#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/type_macros.h>
#include <ert/res_util/subst_list.hpp>

#include <ert/config/config_parser.hpp>
#include <ert/config/config_content.hpp>
#include <ert/config/config_error.hpp>

#include <ert/job_queue/workflow.hpp>
#include <ert/job_queue/workflow_job.hpp>


  typedef struct ert_workflow_list_struct ert_workflow_list_type;

  workflow_type           *  ert_workflow_list_get_workflow(ert_workflow_list_type * workflow_list , const char * workflow_name );
  workflow_type           *  ert_workflow_list_add_workflow( ert_workflow_list_type * workflow_list , const char * workflow_file , const char * workflow_name);
  void                       ert_workflow_list_free( ert_workflow_list_type * workflow_list );
  ert_workflow_list_type  *  ert_workflow_list_alloc_empty( const subst_list_type * subst_list );
  ert_workflow_list_type  *  ert_workflow_list_alloc_load_site_config(const subst_list_type *);
  ert_workflow_list_type  *  ert_workflow_list_alloc(const subst_list_type * context, const config_content_type * config_content);
  ert_workflow_list_type  *  ert_workflow_list_alloc_full(const subst_list_type * context, workflow_joblist_type * workflow_joblist);

  void                       ert_workflow_list_add_jobs_in_directory( ert_workflow_list_type * workflow_list , const char * path );
  void                       ert_workflow_list_add_job( ert_workflow_list_type * workflow_list , const char * job_name , const char * config_file );
  bool                       ert_workflow_list_has_job( const ert_workflow_list_type * workflow_list , const char * job_name);
  const workflow_job_type *  ert_workflow_list_get_job( const ert_workflow_list_type * workflow_list , const char * job_name);
  stringlist_type *          ert_workflow_list_get_job_names(const ert_workflow_list_type * workflow_list);
  void                       ert_workflow_list_add_alias( ert_workflow_list_type * workflow_list , const char * real_name , const char * alias);
  void                       ert_workflow_list_add_config_items( config_parser_type * config );
  bool                       ert_workflow_list_run_workflow__(ert_workflow_list_type * workflow_list, workflow_type * workflow, bool verbose , void * self);
  bool                       ert_workflow_list_has_workflow(ert_workflow_list_type * workflow_list , const char * workflow_name );
  stringlist_type          * ert_workflow_list_alloc_namelist( ert_workflow_list_type * workflow_list );
  void                       ert_workflow_list_set_verbose( ert_workflow_list_type * workflow_list , bool verbose);
  bool                       ert_workflow_list_run_workflow_blocking(ert_workflow_list_type * workflow_list  , const char * workflow_name , void * self);
  const subst_list_type *          ert_workflow_list_get_context(const ert_workflow_list_type * workflow_list);
  int                        ert_workflow_list_get_size( const ert_workflow_list_type * workflow_list);


  UTIL_IS_INSTANCE_HEADER( ert_workflow_list );

#ifdef __cplusplus
}
#endif

#endif
