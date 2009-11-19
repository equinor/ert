#ifndef __SCHED_KW_COMPDAT_H__
#define __SCHED_KW_COMPDAT_H__
#include <stdio.h>
#include <set.h>
#include <sched_macros.h>

typedef struct sched_kw_compdat_struct sched_kw_compdat_type;

sched_kw_compdat_type * sched_kw_compdat_alloc();
void                    sched_kw_compdat_free(sched_kw_compdat_type * );
void                    sched_kw_compdat_fprintf(const sched_kw_compdat_type * , FILE *);
sched_kw_compdat_type * sched_kw_compdat_fread_alloc(FILE *stream);
void                    sched_kw_compdat_fwrite(const sched_kw_compdat_type * , FILE *stream);


KW_HEADER(compdat)


#endif
