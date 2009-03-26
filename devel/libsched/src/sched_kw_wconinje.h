#ifndef __SCHED_KW_WCONINJE_H__
#define __SCHED_KW_WCONINJE_H__


#ifdef __cplusplus
extern "C" {
#endif
#include <sched_macros.h>


typedef struct sched_kw_wconinje_struct sched_kw_wconinje_type;


char ** sched_kw_wconinje_alloc_wells_copy( const sched_kw_wconinje_type * , int * );

void    sched_kw_wconinje_set_surface_flow( const sched_kw_wconinje_type * kw , const char * well, double surface_flow);
void    sched_kw_wconinje_scale_surface_flow( const sched_kw_wconinje_type * kw , const char * well, double factor);
double  sched_kw_wconinje_get_surface_flow( const sched_kw_wconinje_type * kw , const char * well);
bool    sched_kw_wconinje_has_well( const sched_kw_wconinje_type * , const char * );
sched_kw_wconinje_type * sched_kw_wconinje_safe_cast( void * arg );

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
