#ifndef __SCHED_KW_UNTYPED_H__
#define __SCHED_KW_UNTYPED_H__

#include <stdbool.h>

typedef struct sched_kw_untyped_struct sched_kw_untyped_type;

void                    sched_kw_untyped_fprintf(const sched_kw_untyped_type *, FILE *);
sched_kw_untyped_type * sched_kw_untyped_alloc(const char * , bool);
void                    sched_kw_untyped_add_line(sched_kw_untyped_type * , const char *);
void                    sched_kw_untyped_free(sched_kw_untyped_type * );
sched_kw_untyped_type * sched_kw_untyped_fread_alloc(FILE *);
void                    sched_kw_untyped_fwrite(const sched_kw_untyped_type * , FILE *);
#endif
