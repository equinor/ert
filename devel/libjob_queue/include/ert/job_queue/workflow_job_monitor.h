/*
   Copyright (C) 2014  Statoil ASA, Norway.
    
   The file 'workflow_job_monitor.h' is part of ERT - Ensemble based Reservoir Tool.
    
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
#ifndef __WORKFLOW_JOB_MONITOR_H__
#define __WORKFLOW_JOB_MONITOR_H__

#ifdef __cplusplus
extern "C" {
#endif
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>


typedef struct workflow_job_monitor_struct workflow_job_monitor_type;

workflow_job_monitor_type * workflow_job_monitor_alloc();
void workflow_job_monitor_free(workflow_job_monitor_type *monitor);

void workflow_job_monitor_set_pid(workflow_job_monitor_type * monitor, pid_t pid);
pid_t workflow_job_monitor_get_pid(const workflow_job_monitor_type * monitor);

bool workflow_job_monitor_get_blocking(const workflow_job_monitor_type * monitor);
void workflow_job_monitor_set_blocking(workflow_job_monitor_type * monitor, bool blocking);

#ifdef __cplusplus
}
#endif

#endif
