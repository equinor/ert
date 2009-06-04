#ifndef __SCHED_KW_WCONPROD_H__
#define __SCHED_KW_WCONPROD_H__


#ifdef __cplusplus
extern "C" {
#endif
#include <sched_macros.h>


typedef struct sched_kw_wconprod_struct sched_kw_wconprod_type;


char ** sched_kw_wconprod_alloc_wells_copy( const sched_kw_wconprod_type * , int * );

/*******************************************************************/



KW_HEADER(wconprod)

#ifdef __cplusplus
}
#endif
#endif
