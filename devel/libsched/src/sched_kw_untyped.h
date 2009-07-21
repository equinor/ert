#ifndef __SCHED_KW_UNTYPED_H__
#define __SCHED_KW_UNTYPED_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <sched_macros.h>

typedef struct sched_kw_untyped_struct sched_kw_untyped_type;

sched_kw_untyped_type * sched_kw_untyped_alloc(const char * );
sched_kw_untyped_type * sched_kw_untyped_fscanf_alloc(FILE * , bool *, const char *);
void                    sched_kw_untyped_fprintf(const sched_kw_untyped_type *, FILE *);
void                    sched_kw_untyped_free(sched_kw_untyped_type * );
sched_kw_untyped_type * sched_kw_untyped_fread_alloc(FILE *);
void                    sched_kw_untyped_fwrite(const sched_kw_untyped_type * , FILE *);
void                    sched_kw_untyped_add_line(sched_kw_untyped_type *  , const char *, bool);
char **                 sched_kw_untyped_iget_entries_alloc(const sched_kw_untyped_type *, int, int *);


/*******************************************************************/


KW_HEADER(untyped)
#ifdef __cplusplus
}
#endif
#endif
