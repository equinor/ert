#ifndef __SCHED_KW_WCONINJ_H__
#define __SCHED_KW_WCONINJ_H__


#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdbool.h>
#include <sched_macros.h>
#include <hash.h>
#include <stringlist.h>


typedef struct sched_kw_wconinj_struct sched_kw_wconinj_type;


char ** sched_kw_wconinj_alloc_wells_copy( const sched_kw_wconinj_type * , int * );

/*******************************************************************/

KW_HEADER(wconinj)

#ifdef __cplusplus
}
#endif
#endif
