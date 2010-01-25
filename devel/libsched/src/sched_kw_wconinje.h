#ifndef __SCHED_KW_WCONINJE_H__
#define __SCHED_KW_WCONINJE_H__


#ifdef __cplusplus
extern "C" {
#endif
#include <sched_macros.h>
#include <sched_types.h>
#include <stringlist.h>
#include <stdbool.h>
#include <buffer.h>

typedef struct sched_kw_wconinje_struct sched_kw_wconinje_type;
typedef struct wconinje_state_struct    wconinje_state_type;


sched_phase_enum         sched_kw_wconinje_get_phase( const sched_kw_wconinje_type * kw , const char * well_name);
bool                     sched_kw_wconinje_well_open( const sched_kw_wconinje_type * kw, const char * well_name);
char **                  sched_kw_wconinje_alloc_wells_copy( const sched_kw_wconinje_type * , int * );

void                     sched_kw_wconinje_set_surface_flow( const sched_kw_wconinje_type * kw , const char * well, double surface_flow);
void                     sched_kw_wconinje_scale_surface_flow( const sched_kw_wconinje_type * kw , const char * well, double factor);
double                   sched_kw_wconinje_get_surface_flow( const sched_kw_wconinje_type * kw , const char * well);
bool                     sched_kw_wconinje_has_well( const sched_kw_wconinje_type * , const char * );
sched_kw_wconinje_type * sched_kw_wconinje_safe_cast( void * arg );
void                     sched_kw_wconinje_shift_surface_flow( const sched_kw_wconinje_type * kw , const char * well_name , double delta_surface_flow);
bool                     sched_kw_wconinje_buffer_fwrite( const sched_kw_wconinje_type * kw , const char * well_name , buffer_type * buffer);


void                     wconinje_state_free__( void * arg );
wconinje_state_type    * wconinje_state_alloc( );
void                     wconinje_state_free( wconinje_state_type * wconinje );

/*******************************************************************/



KW_HEADER(wconinje)



#ifdef __cplusplus
}
#endif
#endif
