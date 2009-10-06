#ifndef __SHCED_KW_H__
#define __SCHED_KW_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <hash.h>

typedef enum {WCONHIST =  0, 
              DATES    =  1, 
              COMPDAT  =  2, 
              TSTEP    =  3, 
              TIME     =  4,   /* Not implemented support */
              WELSPECS =  5, 
              GRUPTREE =  6,
              INCLUDE  =  7,   
              UNTYPED  =  8,
              WCONINJ  =  9,
              WCONINJE = 10, 
              WCONINJH = 11,
              WCONPROD = 12,
              NUM_SCHED_KW_TYPES = 13} sched_type_enum;
              


#define WCONHIST_STRING  "WCONHIST"
#define DATES_STRING     "DATES"
#define COMPDAT_STRING   "COMPDAT"
#define TSTEP_STRING     "TSTEP"
#define TIME_STRING      "TIME"
#define WELSPECS_STRING  "WELSPECS"
#define GRUPTREE_STRING  "GRUPTREE"
#define INCLUDE_STRING   "INCLUDE"
#define WCONINJ_STRING   "WCONINJ"
#define WCONINJE_STRING  "WCONINJE"
#define WCONINJH_STRING  "WCONINJH"
#define WCONPROD_STRING  "WCONPROD"

#define UNTYPED_STRING   "UNTYPED"
              
typedef struct sched_kw_struct sched_kw_type;



/*****************************************************************/



void sched_kw_free__(void *);
const char     * sched_kw_get_type_name( const sched_kw_type * sched_kw );
sched_type_enum  sched_kw_get_type(const sched_kw_type *);
sched_kw_type *  sched_kw_token_alloc(const stringlist_type * tokens, int * token_index, hash_type * fixed_length_table);
void             sched_kw_fprintf(const sched_kw_type *, FILE *);
void             sched_kw_free(sched_kw_type *);

sched_kw_type  * sched_kw_alloc_copy(const sched_kw_type * );
sched_kw_type ** sched_kw_restart_file_split_alloc(const sched_kw_type *, int *);
time_t           sched_kw_get_new_time(const sched_kw_type *, time_t);
char          ** sched_kw_alloc_well_list(const sched_kw_type *, int *);
hash_type      * sched_kw_alloc_well_obs_hash(const sched_kw_type *);
void             sched_kw_alloc_child_parent_list(const sched_kw_type *, char ***, char ***, int *);
void           * sched_kw_get_data( sched_kw_type * kw);
void             sched_kw_set_restart_nr( sched_kw_type * kw , int restart_nr);

#ifdef __cplusplus
}
#endif
#endif
