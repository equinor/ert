#ifndef __SCHED_KW_WCONINJH_H__
#define __SCHED_KW_WCONINJH_H__



#include <stdio.h>
#include <stdbool.h>
#include <sched_macros.h>


typedef struct sched_kw_wconhist_struct sched_kw_wconinjh_type;

sched_kw_wconinjh_type * sched_kw_wconinjh_fscanf_alloc( FILE *, bool *, const char *);
void                     sched_kw_wconinjh_free(sched_kw_wconinjh_type * );
void                     sched_kw_wconinjh_fprintf(const sched_kw_wconinjh_type * , FILE *);
void                     sched_kw_wconinjh_fwrite(const sched_kw_wconinjh_type *, FILE *);
sched_kw_wconinjh_type * sched_kw_wconinjh_fread_alloc( FILE *);

hash_type * sched_kw_wconinjh_alloc_well_obs_hash(const sched_kw_wconinjh_type *);

/*******************************************************************/

KW_FSCANF_ALLOC_HEADER(wconinjh)
KW_FWRITE_HEADER(wconinjh)
KW_FREAD_ALLOC_HEADER(wconinjh)
KW_FREE_HEADER(wconinjh)
KW_FPRINTF_HEADER(wconinjh)


#endif
