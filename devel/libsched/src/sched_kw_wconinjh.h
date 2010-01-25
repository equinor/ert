#ifndef __SCHED_KW_WCONINJH_H__
#define __SCHED_KW_WCONINJH_H__
#ifdef __cplusplus
extern "C" {
#endif



#include <stdio.h>
#include <stdbool.h>
#include <sched_macros.h>
#include <hash.h>
#include <stringlist.h>


typedef struct sched_kw_wconhist_struct sched_kw_wconinjh_type;
typedef struct wconinjh_state_struct    wconinjh_state_type;

sched_kw_wconinjh_type * sched_kw_wconinjh_fscanf_alloc( FILE *, bool *, const char *);
void                     sched_kw_wconinjh_free(sched_kw_wconinjh_type * );
void                     sched_kw_wconinjh_fprintf(const sched_kw_wconinjh_type * , FILE *);
void                     sched_kw_wconinjh_fwrite(const sched_kw_wconinjh_type *, FILE *);
sched_kw_wconinjh_type * sched_kw_wconinjh_fread_alloc( FILE *);

hash_type * sched_kw_wconinjh_alloc_well_obs_hash(const sched_kw_wconinjh_type *);

void                     sched_kw_wconinjh_init_well_list( const sched_kw_wconinjh_type * kw , stringlist_type * well_list);
void                     sched_kw_wconinjh_update_state( const sched_kw_wconinjh_type * kw , wconinjh_state_type * state , const char * well_name , int report_step );
void                     wconinjh_state_free__( void * arg );
wconinjh_state_type    * wconinjh_state_alloc( );
void                     wconinjh_state_free( wconinjh_state_type * wconinjh );


/*******************************************************************/
KW_HEADER(wconinjh)

#ifdef __cplusplus
}
#endif
#endif
