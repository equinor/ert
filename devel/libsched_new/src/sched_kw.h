#ifndef __SHCED_KW_H__
#define __SCHED_KW_H__
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <hash.h>

typedef enum {WCONHIST, DATES , COMPDAT , TSTEP, TIME, WELSPECS, GRUPTREE,
              INCLUDE, RPTSCHED, DRSDT, SKIPREST, RPTRST, TUNING, WHISTCTL, UNTYPED, 
              WCONINJ, WCONINJE, WCONINJH, WCONPROD
              } sched_type_enum;

#define WCONHIST_STRING  "WCONHIST"
#define DATES_STRING     "DATES"
#define COMPDAT_STRING   "COMPDAT"
#define TSTEP_STRING     "TSTEP"
#define TIME_STRING      "TIME"
#define WELSPECS_STRING  "WELSPECS"
#define GRUPTREE_STRING  "GRUPTREE"
#define INCLUDE_STRING   "INCLUDE"
#define RPTSCHED_STRING  "RPTSCHED"
#define DRSDT_STRING     "DRSDT"
#define SKIPREST_STRING  "SKIPREST"
#define RPTRST_STRING    "RPTRST"
#define TUNING_STRING    "TUNING"
#define WHISTCTL_STRING  "WHISTCTL"
#define WCONINJ_STRING   "WCONINJ"
#define WCONINJE_STRING  "WCONINJE"
#define WCONINJH_STRING  "WCONINJH"
#define WCONPROD_STRING  "WCONPROD"
              
typedef struct sched_kw_struct sched_kw_type;



/*****************************************************************/



void sched_kw_free__(void *);
sched_type_enum  sched_kw_get_type(const sched_kw_type *);
sched_kw_type *  sched_kw_fscanf_alloc(FILE *, bool *);
void             sched_kw_fprintf(const sched_kw_type *, FILE *);
void             sched_kw_free(sched_kw_type *);
void             sched_kw_fwrite(sched_kw_type *, FILE *);
sched_kw_type *  sched_kw_fread_alloc(FILE *, bool * at_eof);

sched_kw_type ** sched_kw_restart_file_split_alloc(const sched_kw_type *, int *);
time_t           sched_kw_get_new_time(const sched_kw_type *, time_t);
char          ** sched_kw_alloc_well_list(const sched_kw_type *, int *);
hash_type      * sched_kw_alloc_well_obs_hash(const sched_kw_type *);
void             sched_kw_alloc_child_parent_list(const sched_kw_type *, char ***, char ***, int *);
#endif
