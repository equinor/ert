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


#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/subst_list.h>
#include <ert/util/log.h>

#include <ert/config/config.h>
#include <ert/config/config_error.h>

#include <ert/job_queue/workflow.h>
#include <ert/enkf/ert_workflow_list.h>


typedef struct ert_workflow_list_handler_data_struct ert_workflow_list_handler_data_type;

void ert_workflow_list_handler_free(ert_workflow_list_handler_data_type *tdata);
void ert_workflow_list_handler_set_pointer(ert_workflow_list_handler_data_type * tdata, void * self);
void ert_workflow_list_handler_set_workflow_name(ert_workflow_list_handler_data_type * tdata, char *workflow_name);
void ert_workflow_list_handler_set_workflow_list(ert_workflow_list_handler_data_type * tdata, ert_workflow_list_type * workflow_list);
ert_workflow_list_handler_data_type * ert_workflow_list_handler_alloc();
void ert_workflow_list_handler_run_workflow(ert_workflow_list_handler_data_type * tdata, ert_workflow_list_type * workflow_list,char *workflow_name, void * self);
bool ert_workflow_list_handler_read_result(ert_workflow_list_handler_data_type *tdata);
bool ert_workflow_list_handler_is_running(ert_workflow_list_handler_data_type *tdata);
void ert_workflow_list_handler_stop_workflow(ert_workflow_list_handler_data_type *tdata);
void ert_workflow_list_handler_join_workflow(ert_workflow_list_handler_data_type *tdata);
bool ert_workflow_list_handler_is_killed(ert_workflow_list_handler_data_type *tdata);

#ifdef __cplusplus
}
#endif


