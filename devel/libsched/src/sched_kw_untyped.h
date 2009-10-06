#ifndef __SCHED_KW_UNTYPED_H__
#define __SCHED_KW_UNTYPED_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <sched_macros.h>
#include <stringlist.h>

typedef struct sched_kw_untyped_struct sched_kw_untyped_type;

sched_kw_untyped_type * sched_kw_untyped_alloc(const stringlist_type * tokens , int * token_index , int rec_len);
sched_kw_untyped_type * sched_kw_untyped_alloc_empty(const char * , int rec_len);
void                    sched_kw_untyped_fprintf(const sched_kw_untyped_type *, FILE *);
void                    sched_kw_untyped_free(sched_kw_untyped_type * );
sched_kw_untyped_type * sched_kw_untyped_fread_alloc(FILE *);
void                    sched_kw_untyped_fwrite(const sched_kw_untyped_type * , FILE *);
void                    sched_kw_untyped_add_line(sched_kw_untyped_type *  , const char *, bool);
void                    sched_kw_untyped_add_tokens( sched_kw_untyped_type * kw , const stringlist_type * tokens);
char **                 sched_kw_untyped_iget_entries_alloc(const sched_kw_untyped_type *, int, int *);

/*******************************************************************/

KW_FREE_HEADER(untyped)         
KW_FPRINTF_HEADER(untyped)   
KW_COPYC_HEADER(untyped)

#ifdef __cplusplus
}
#endif
#endif
