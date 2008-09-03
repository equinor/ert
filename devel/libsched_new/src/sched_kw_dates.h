#ifndef __SCHED_KW_DATES__
#define __SCHED_KW_DATES__
#include <stdbool.h>
#include <time.h>
#include <sched_macros.h>

typedef struct sched_kw_dates_struct sched_kw_dates_type;

sched_kw_dates_type  * sched_kw_dates_fscanf_alloc(FILE * , bool *, const char * );
void sched_kw_dates_fprintf(const sched_kw_dates_type * , FILE *);
void sched_kw_dates_free(sched_kw_dates_type * );
void sched_kw_dates_fwrite(const sched_kw_dates_type * , FILE * );
sched_kw_dates_type * sched_kw_dates_fread_alloc(FILE * );

int                   sched_kw_dates_get_size(const sched_kw_dates_type *);
sched_kw_dates_type * sched_kw_dates_alloc_from_time_t(time_t );
time_t                sched_kw_dates_iget_time_t(const sched_kw_dates_type *, int);
time_t                sched_kw_dates_get_time_t(const sched_kw_dates_type *);

/*******************************************************************/



KW_FSCANF_ALLOC_HEADER(dates)
KW_FWRITE_HEADER(dates)
KW_FREAD_ALLOC_HEADER(dates)
KW_FREE_HEADER(dates)
KW_FPRINTF_HEADER(dates)

#endif
