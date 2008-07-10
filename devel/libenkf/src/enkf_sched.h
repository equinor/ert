#ifndef __ENKF_SCHED_H__
#define __ENKF_SCHED_H__
#include <stdio.h>


typedef struct enkf_sched_struct      enkf_sched_type;
typedef struct enkf_sched_node_struct enkf_sched_node_type;



void enkf_sched_fprintf(const enkf_sched_type *  , FILE * );

#endif
