#ifndef __SCHED_KW_TSTEP__
#define __SCHED_KW_TSTEP__
#include <hash.h>
#include <time.h>
#include <sched_macros.h>

typedef struct sched_kw_tstep_struct sched_kw_tstep_type;

sched_kw_tstep_type * sched_kw_tstep_fscanf_alloc(FILE *, bool *, const char *);
void                  sched_kw_tstep_free(sched_kw_tstep_type * );
void                  sched_kw_tstep_fprintf(const sched_kw_tstep_type *, FILE *);
void                  sched_kw_tstep_fwrite(const sched_kw_tstep_type * , FILE *);
sched_kw_tstep_type * sched_kw_tstep_fread_alloc(FILE *);

int                   sched_kw_tstep_get_size(const sched_kw_tstep_type *);
sched_kw_tstep_type * sched_kw_tstep_alloc_from_double(double);
double                sched_kw_tstep_iget_step(const sched_kw_tstep_type *, int);
double                sched_kw_tstep_get_step(const sched_kw_tstep_type *);
time_t                sched_kw_tstep_get_new_time(const sched_kw_tstep_type *, time_t);



/*******************************************************************/



KW_FSCANF_ALLOC_HEADER(tstep)
KW_FWRITE_HEADER(tstep)
KW_FREAD_ALLOC_HEADER(tstep)
KW_FREE_HEADER(tstep)
KW_FPRINTF_HEADER(tstep)
#endif
