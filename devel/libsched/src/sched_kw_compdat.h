#ifndef __SCHED_KW_COMPDAT_H__
#define __SCHED_KW_COMPDAT_H__
#include <ecl_kw.h>
#include <stdio.h>
#include <set.h>


typedef struct sched_kw_compdat_struct sched_kw_compdat_type;

void                    sched_kw_compdat_set_conn_factor(sched_kw_compdat_type *  , const float *, const float * , const int *  , const int * );
void                    sched_kw_compdat_init_conn_factor(sched_kw_compdat_type * , const ecl_kw_type *, const ecl_kw_type * , const int * , const int * , bool *);
sched_kw_compdat_type * sched_kw_compdat_alloc();
void                    sched_kw_compdat_free(sched_kw_compdat_type * );
void                    sched_kw_compdat_add_line(sched_kw_compdat_type * kw , const char *);
void                    sched_kw_compdat_fprintf(const sched_kw_compdat_type * , FILE *);
sched_kw_compdat_type * sched_kw_compdat_fread_alloc(FILE *stream);
void                    sched_kw_compdat_fwrite(const sched_kw_compdat_type * , FILE *stream);
void                    sched_kw_compdat_update_well_set(const sched_kw_compdat_type * , set_type * );
#endif
