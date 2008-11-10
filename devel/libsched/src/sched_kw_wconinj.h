#ifndef __SCHED_KW_WCONINJ_H__
#define __SCHED_KW_WCONINJ_H__


#ifdef __cplusplus
extern "C" {
#endif
#include <sched_macros.h>


typedef struct sched_kw_wconinj_struct sched_kw_wconinj_type;


char ** sched_kw_wconinj_alloc_wells_copy( const sched_kw_wconinj_type * , int * );

/*******************************************************************/



KW_FSCANF_ALLOC_HEADER(wconinj)
KW_FWRITE_HEADER(wconinj)
KW_FREAD_ALLOC_HEADER(wconinj)
KW_FREE_HEADER(wconinj)
KW_FPRINTF_HEADER(wconinj)



#ifdef __cplusplus
}
#endif
#endif
