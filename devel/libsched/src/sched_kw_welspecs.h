#ifndef __SCHED_KW_WELSPECS_H__
#define __SCHED_KW_WELSPECS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdbool.h>
#include <sched_macros.h>
#include <stringlist.h>


/*************************************************************/

typedef struct sched_kw_welspecs_struct sched_kw_welspecs_type;



sched_kw_welspecs_type * sched_kw_welspecs_fscanf_alloc(FILE *, bool *, const char *);
sched_kw_welspecs_type * sched_kw_welspecs_fread_alloc(FILE *);
void sched_kw_welspecs_free(sched_kw_welspecs_type *);
void sched_kw_welspecs_fprintf(const sched_kw_welspecs_type *, FILE *);
void sched_kw_welspecs_fwrite(const sched_kw_welspecs_type *, FILE *);

void sched_kw_welspecs_alloc_child_parent_list(const sched_kw_welspecs_type *, char ***, char ***, int *);

/*******************************************************************/



KW_HEADER(welspecs)

#ifdef __cplusplus
}
#endif
#endif
