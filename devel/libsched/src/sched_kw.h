#ifndef __SHCED_KW_H__
#define __SCHED_KW_H__
#include <stdlib.h>
#include <hash.h>
#include <stdbool.h>
#include <time.h>
#include <sched_kw_dates.h>
#include <history.h>

typedef enum {WCONHIST=0 , UNTYPED , DATES , COMPDAT , TSTEP} sched_type_enum;

typedef void  (add_line_ftype)  (void * , int , const char **);
typedef void  (free_ftype)      (void *);

typedef struct sched_kw_struct sched_kw_type;

void            * sched_kw_get_data_ref(const sched_kw_type * );
sched_type_enum   sched_kw_get_type(const sched_kw_type * );
sched_kw_type   * sched_kw_alloc(const char * , sched_type_enum , bool , int * , double * , const time_t *);
void              sched_kw_free(sched_kw_type * );
void              sched_kw_free__(void * );
void              sched_kw_fprintf(const sched_kw_type * kw , int , time_t , double , FILE *, bool *);
void              sched_kw_add_line(sched_kw_type * , const char * , const time_t * , const hash_type * , bool *);
sched_kw_type   * sched_kw_fread_alloc(int *, double * , const time_t * , int , time_t , double , FILE * , bool * , bool *);
void              sched_kw_fwrite(const sched_kw_type *kw , FILE *);
void              sched_kw_fprintf_rates(const sched_kw_type * , const char *, const char * , date_node_type **);
void              sched_kw_fprintf_days_dat(const sched_kw_type * , FILE *);
void              sched_kw_get_report_step(const sched_kw_type *  , time_t , int * );
void              sched_kw_make_history(const sched_kw_type *  , history_type * , date_node_type **);
#endif
