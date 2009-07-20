#ifndef __SCHED_KW_INCLUDE_H__
#define __SCHED_KW_INCLDUE_H__

#include <sched_macros.h>

typedef struct sched_kw_include_struct sched_kw_include_type;


sched_kw_include_type  * sched_kw_include_fscanf_alloc( FILE *, bool *, const char *);
void                     sched_kw_include_free(sched_kw_include_type * );
void                     sched_kw_include_fprintf(const sched_kw_include_type * , FILE *);
void                     sched_kw_include_fwrite(const sched_kw_include_type *, FILE *);
sched_kw_include_type  * sched_kw_include_fread_alloc( FILE *);

KW_HEADER(include)

#endif
