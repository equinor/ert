/*
   Copyright (C) 2014  Statoil ASA, Norway.
    
   The file 'ert_workflow_list_handler.c' is part of ERT - Ensemble based Reservoir Tool.
    
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

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>

#include <ert/util/hash.h>
#include <ert/util/stringlist.h>
#include <ert/util/util.h>
#include <ert/util/subst_list.h>

#include <ert/config/config.h>
#include <ert/config/config_error.h>
#include <ert/config/config_schema_item.h>

#include <ert/job_queue/workflow.h>
#include <ert/job_queue/workflow_job.h>
#include <ert/job_queue/workflow_joblist.h>

#include <ert/enkf/ert_workflow_list.h>
#include <ert/enkf/config_keys.h>
#include <ert/enkf/enkf_defaults.h>
#include <ert/enkf/ert_workflow_list_handler.h>
#include <pthread.h>
#include <ert/job_queue/workflow_job_monitor.h>


struct ert_workflow_list_handler_data_struct{
     ert_workflow_list_type      * workflow_list;
     const char                  * workflow_name;
     void                        * self;
     workflow_job_monitor_type   * monitor;
     bool                          result;
     bool                          running;
     bool                          killed;
     pthread_t                     tid;
};

void * ert_workflow_list_handler_workflowthread(void *arg){
      int oldState = NULL;
      pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &oldState);
      ert_workflow_list_handler_data_type *tdata = (ert_workflow_list_handler_data_type *)arg;
      tdata->result = ert_workflow_list_run_workflow(tdata->workflow_list, tdata->monitor, tdata->workflow_name, tdata->self);
      tdata->running = false;
}

void ert_workflow_list_handler_run_workflow(ert_workflow_list_handler_data_type * tdata, ert_workflow_list_type * workflow_list,char *workflow_name, void * self){
        pthread_t tid;
        tdata->running = true;
        tdata->workflow_list = workflow_list;
        tdata->workflow_name = workflow_name;
        tdata->self = self;
        workflow_job_monitor_type * monitor = tdata->monitor;
        workflow_job_monitor_set_blocking(monitor, false);
        int rc = pthread_create(&tid, NULL, ert_workflow_list_handler_workflowthread, (void *)tdata);
        if(rc)			/* could not create thread */
        {
                printf("\n ERROR: return code from pthread_create is %d \n", rc);
        }
        tdata->tid = tid;
}

void ert_workflow_list_handler_set_pointer(ert_workflow_list_handler_data_type * tdata, void * self){
    tdata->self = self;
}

void ert_workflow_list_handler_set_workflow_name(ert_workflow_list_handler_data_type * tdata, char *workflow_name){
  tdata->workflow_name = workflow_name;
}

void ert_workflow_list_handler_set_workflow_list(ert_workflow_list_handler_data_type * tdata, ert_workflow_list_type * workflow_list){
  tdata->workflow_list = workflow_list;
}

void ert_workflow_list_handler_free(ert_workflow_list_handler_data_type *tdata){
  workflow_job_monitor_free(tdata->monitor);
  free(tdata);
}

ert_workflow_list_handler_data_type * ert_workflow_list_handler_alloc(){
  ert_workflow_list_handler_data_type * tdata = util_malloc( sizeof * tdata );
  workflow_job_monitor_type * monitor = workflow_job_monitor_alloc();
  tdata->monitor = monitor;
  tdata->killed = false;
  return tdata;
}

void ert_workflow_list_handler_stop_workflow(ert_workflow_list_handler_data_type *tdata){
  if(tdata->running){
      tdata->running = false;
      pid_t thread_pid = workflow_job_monitor_get_pid(tdata->monitor);
      pid_t handler_pid = getpid();
      if (thread_pid == handler_pid){
         pthread_cancel(tdata->tid);
      }else{
          pid_t pid = workflow_job_monitor_get_pid(tdata->monitor);
          kill(pid, SIGTERM);
          sleep(2);
          kill(pid, SIGKILL);
      }
      tdata->result = false;
      tdata->killed = true;
  }

}

void ert_workflow_list_handler_join_workflow(ert_workflow_list_handler_data_type *tdata){
  pthread_join(tdata->tid, NULL);
}

bool ert_workflow_list_handler_is_running(ert_workflow_list_handler_data_type *tdata){
  return tdata->running;
}

bool ert_workflow_list_handler_is_killed(ert_workflow_list_handler_data_type *tdata){
  return tdata->killed;
}

bool ert_workflow_list_handler_read_result(ert_workflow_list_handler_data_type *tdata){
  return tdata->result;
}


