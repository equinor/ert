#ifndef __SCHED_KW_DATES__
#define __SCHED_KW_DATES__
#include <hash.h>
#include <stdbool.h>
#include <history.h>
#include "date_node.h"

typedef struct sched_kw_dates_struct sched_kw_dates_type;

void                  sched_kw_dates_fprintf(const sched_kw_dates_type *, FILE *, int , time_t , bool *);
void                  sched_kw_dates_add_line(sched_kw_dates_type *, const time_t * , const char *, const hash_type * , bool);
sched_kw_dates_type * sched_kw_dates_alloc(int * , const time_t *);
void                  sched_kw_dates_free(sched_kw_dates_type * );
sched_kw_dates_type * sched_kw_dates_fread_alloc(int *  , const time_t * , int , time_t , FILE * , bool *);
void                  sched_kw_dates_fwrite(const sched_kw_dates_type * , FILE *);
void                  sched_kw_dates_iterate_current(const sched_kw_dates_type * , date_node_type **);
void                  sched_kw_dates_fprintf_days_dat(const sched_kw_dates_type * , FILE *);
void                  sched_kw_dates_get_report_step(const sched_kw_dates_type * , time_t , int * );
void                  sched_kw_dates_make_history(const sched_kw_dates_type * ,  history_type * );
#endif
