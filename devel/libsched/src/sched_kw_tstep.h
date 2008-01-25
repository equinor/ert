#ifndef __SCHED_KW_TSTEP__
#define __SCHED_KW_TSTEP__
#include <hash.h>
#include <time.h>

typedef struct sched_kw_tstep_struct sched_kw_tstep_type;

void                  sched_kw_tstep_fprintf(const sched_kw_tstep_type *, FILE *, int , time_t , bool *);
void                  sched_kw_tstep_add_line(sched_kw_tstep_type *, const char * , bool *);
sched_kw_tstep_type * sched_kw_tstep_alloc(int * , double * , const time_t *);
void                  sched_kw_tstep_free(sched_kw_tstep_type * );
sched_kw_tstep_type * sched_kw_tstep_fread_alloc(int *  , double * , const time_t *, int , time_t , FILE * , bool *);
void                  sched_kw_tstep_fwrite(const sched_kw_tstep_type * , FILE *);
void                  sched_kw_tstep_iterate_current(const sched_kw_tstep_type * , int *);
#endif
