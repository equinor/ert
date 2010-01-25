#ifndef __SCHED_KW_WCONHIST_H__
#define __SCHED_KW_WCONHIST_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <sched_macros.h>
#include <hash.h>
#include <stringlist.h>
#include <sched_types.h>
  
typedef  struct  sched_kw_wconhist_struct sched_kw_wconhist_type;
typedef  struct  wconhist_state_struct    wconhist_state_type;  

#define WCONHIST_DEFAULT_STATUS  OPEN

sched_kw_wconhist_type * sched_kw_wconhist_fscanf_alloc( FILE *, bool *, const char *);
void                     sched_kw_wconhist_free(sched_kw_wconhist_type * );
void                     sched_kw_wconhist_fprintf(const sched_kw_wconhist_type * , FILE *);
void                     sched_kw_wconhist_fwrite(const sched_kw_wconhist_type *, FILE *);
sched_kw_wconhist_type * sched_kw_wconhist_fread_alloc( FILE *);
hash_type              * sched_kw_wconhist_alloc_well_obs_hash(const sched_kw_wconhist_type *);
double 	 		 sched_kw_wconhist_get_orat( sched_kw_wconhist_type * kw , const char * well_name);
void   	 		 sched_kw_wconhist_scale_orat(  sched_kw_wconhist_type * kw , const char * well_name, double factor);
void   	 		 sched_kw_wconhist_set_surface_flow(  sched_kw_wconhist_type * kw , const char * well_name , double orat);
bool   	 		 sched_kw_wconhist_has_well( const sched_kw_wconhist_type * kw , const char * well_name);
bool                     sched_kw_wconhist_well_open( const sched_kw_wconhist_type * kw, const char * well_name);
void                     sched_kw_wconhist_shift_orat( sched_kw_wconhist_type * kw , const char * well_name, double shift_value);
void                     sched_kw_wconhist_shift_grat( sched_kw_wconhist_type * kw , const char * well_name, double shift_value);
void                     sched_kw_wconhist_shift_wrat( sched_kw_wconhist_type * kw , const char * well_name, double shift_value);
void                     sched_kw_wconhist_update_state(const sched_kw_wconhist_type * kw , wconhist_state_type * state , const char * well_name , int report_step );

void                     sched_kw_wconhist_init_well_list( const sched_kw_wconhist_type * kw , stringlist_type * well_list);
void                     wconhist_state_free__( void * arg );
wconhist_state_type    * wconhist_state_alloc( );
void                     wconhist_state_free( wconhist_state_type * wconhist );


UTIL_SAFE_CAST_HEADER( sched_kw_wconhist );
/*******************************************************************/



KW_HEADER(wconhist)

#ifdef __cplusplus
}
#endif
#endif
