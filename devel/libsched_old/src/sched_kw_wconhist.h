#ifndef __SCHED_KW_WCONHIST_H__
#define __SCHED_KW_WCONHIST_H__
#include <history.h>

typedef struct sched_kw_wconhist_struct sched_kw_wconhist_type;


sched_kw_wconhist_type * sched_kw_wconhist_alloc();
void                     sched_kw_wconhist_free(sched_kw_wconhist_type * );
void                     sched_kw_wconhist_add_line(sched_kw_wconhist_type * kw , const char *);
void                     sched_kw_wconhist_fprintf(const sched_kw_wconhist_type * , FILE *);
void                     sched_kw_wconhist_fwrite(const sched_kw_wconhist_type *, FILE *);
sched_kw_wconhist_type * sched_kw_wconhist_fread_alloc( FILE *);
void                     sched_kw_wconhist_fprintf_rates(const sched_kw_wconhist_type * , const char *  , const char * , int );
void                     sched_kw_wconhist_make_history(const sched_kw_wconhist_type * , int , history_type * );
#endif
