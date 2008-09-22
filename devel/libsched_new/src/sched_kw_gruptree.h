#ifndef __SCHED_KW_GRUPTREE_H__
#define __SCHED_KW_GRUPTREE_H__
#include <stdio.h>
#include <sched_macros.h>

typedef struct sched_kw_gruptree_struct sched_kw_gruptree_type;

sched_kw_gruptree_type * sched_kw_gruptree_fscanf_alloc(FILE *, bool *, const char *);
void                     sched_kw_gruptree_free        (sched_kw_gruptree_type *);
void                     sched_kw_gruptree_fprintf     (const sched_kw_gruptree_type *, FILE * );
void                     sched_kw_gruptree_fwrite      (const sched_kw_gruptree_type *, FILE *);
sched_kw_gruptree_type * sched_kw_gruptree_fread_alloc (FILE *);

void sched_kw_gruptree_alloc_child_parent_list(const sched_kw_gruptree_type *, char ***, char ***, int *);

/*******************************************************************/



KW_FSCANF_ALLOC_HEADER(gruptree)
KW_FWRITE_HEADER(gruptree)
KW_FREAD_ALLOC_HEADER(gruptree)
KW_FREE_HEADER(gruptree)
KW_FPRINTF_HEADER(gruptree)


#endif
