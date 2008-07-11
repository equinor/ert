#ifndef __ENKF_SCHED_H__
#define __ENKF_SCHED_H__
#include <stdio.h>
#include <sched_file.h>
#include <ext_joblist.h>
#include <stdlib.h>
#include <stringlist.h>


typedef struct enkf_sched_struct      enkf_sched_type;
typedef struct enkf_sched_node_struct enkf_sched_node_type;



void               	 enkf_sched_fprintf(const enkf_sched_type *  , FILE * );
enkf_sched_type *  	 enkf_sched_fscanf_alloc(const char * , const sched_file_type *  , const ext_joblist_type *, const stringlist_type *);
void               	 enkf_sched_free(enkf_sched_type *);
int                	 enkf_sched_get_num_nodes(const enkf_sched_type *);
int                	 enkf_sched_get_schedule_num_reports(const enkf_sched_type *);
const enkf_sched_node_type *   enkf_sched_iget_node(const enkf_sched_type * , int);
void                     enkf_sched_node_get_data(const enkf_sched_node_type * , int * , int * , int * , bool * , stringlist_type ** );

#endif
