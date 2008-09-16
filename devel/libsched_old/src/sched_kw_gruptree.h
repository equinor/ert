#ifndef __SCHED_KW_GRUPTREE_H__
#define __SCHED_KW_GRUPTREE_H__
#include <stdio.h>

typedef struct sched_kw_gruptree_struct sched_kw_gruptree_type;

sched_kw_gruptree_type * sched_kw_gruptree_alloc();
sched_kw_gruptree_type * sched_kw_gruptree_fread_alloc(FILE *);
void sched_kw_gruptree_free(sched_kw_gruptree_type *);
void sched_kw_gruptree_fprintf(const sched_kw_gruptree_type *, FILE * );
void sched_kw_gruptree_add_line(sched_kw_gruptree_type *, const char *);
void sched_kw_gruptree_fwrite(const sched_kw_gruptree_type *, FILE *);
#endif
