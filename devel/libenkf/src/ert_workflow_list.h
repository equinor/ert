/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
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

#ifndef __ERT_WORKFLOW_LIST_H__
#define __ERT_WORKFLOW_LIST_H__


#ifdef __cplusplus
extern "C" {
#endif

#include <config.h>


  typedef struct ert_workflow_list_struct ert_workflow_list_type;
  
  void                       ert_workflow_list_free( ert_workflow_list_type * workflow_list );
  ert_workflow_list_type  *  ert_workflow_list_alloc();
  void                       ert_workflow_list_add_jobs_in_directory( ert_workflow_list_type * workflow_list , const char * path);
  void                       ert_workflow_list_add_job( ert_workflow_list_type * workflow_list , const char * job_name , const char * config_file );
  void                       ert_workflow_list_update_config( config_type * config );
  void                       ert_workflow_list_init( ert_workflow_list_type * workflow_list , config_type * config );
  void                       ert_workflow_list_add_workflow( ert_workflow_list_type * workflow_list , const char * workflow_file , const char * workflow_name);
  bool                       ert_workflow_list_run_workflow(ert_workflow_list_type * workflow_list  , const char * workflow_name , void * self);
  bool                       ert_workflow_list_has_workflow(ert_workflow_list_type * workflow_list , const char * workflow_name );

#ifdef __cplusplus
}
#endif

#endif
