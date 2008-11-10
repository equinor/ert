#ifndef __SCHED_KW_WCONINJE_H__
#define __SCHED_KW_WCONINJE_H__


#ifdef __cplusplus
extern "C" {
#endif
#include <sched_macros.h>


typedef struct sched_kw_wconinje_struct sched_kw_wconinje_type;


char ** sched_kw_wconinje_alloc_wells_copy( const sched_kw_wconinje_type * , int * );

/*******************************************************************/



KW_FSCANF_ALLOC_HEADER(wconinje)
KW_FWRITE_HEADER(wconinje)
KW_FREAD_ALLOC_HEADER(wconinje)
KW_FREE_HEADER(wconinje)
KW_FPRINTF_HEADER(wconinje)



#ifdef __cplusplus
}
#endif
#endif
