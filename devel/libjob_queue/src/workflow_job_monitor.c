/*
   Copyright (C) 2014  Statoil ASA, Norway.
    
   The file 'workflow_job_monitor.c' is part of ERT - Ensemble based Reservoir Tool.
    
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

#include <ert/job_queue/workflow_job_monitor.h>
#include <ert/util/util.h>

struct workflow_job_monitor_struct{
    pid_t   processid;
    bool    blocking;
};

workflow_job_monitor_type * workflow_job_monitor_alloc(){
    workflow_job_monitor_type * monitor = util_malloc( sizeof * monitor);
    monitor->blocking = true;
    return monitor;
}

void workflow_job_monitor_free(workflow_job_monitor_type *monitor){
    free(monitor);
}

void workflow_job_monitor_set_pid(workflow_job_monitor_type * monitor, pid_t pid){
    monitor->processid = pid;
}

pid_t workflow_job_monitor_get_pid(workflow_job_monitor_type * monitor){
    return monitor->processid;
}

bool workflow_job_monitor_get_blocking(workflow_job_monitor_type * monitor){
    return monitor->blocking;
}

void workflow_job_monitor_set_blocking(workflow_job_monitor_type * monitor, bool blocking){
    monitor->blocking = blocking;
}

