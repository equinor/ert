#ifndef __SCHED_KW_WELSPECS_H__
#define __SCHED_KW_WELSPECS_H__
#include <stdio.h>
#include <stdbool.h>
#include <sched_macros.h>



/*************************************************************/

typedef struct sched_kw_welspecs_struct sched_kw_welspecs_type;

sched_kw_welspecs_type * sched_kw_welspecs_fscanf_alloc(FILE *, bool *, const char *);
sched_kw_welspecs_type * sched_kw_welspecs_fread_alloc(FILE *);
void sched_kw_welspecs_free(sched_kw_welspecs_type *);
void sched_kw_welspecs_fprintf(const sched_kw_welspecs_type *, FILE *);
void sched_kw_welspecs_fwrite(const sched_kw_welspecs_type *, FILE *);



/*******************************************************************/



KW_FSCANF_ALLOC_HEADER(welspecs)
KW_FWRITE_HEADER(welspecs)
KW_FREAD_ALLOC_HEADER(welspecs)
KW_FREE_HEADER(welspecs)
KW_FPRINTF_HEADER(welspecs)


#endif
