#ifndef __SCHED_UTIL_H__
#define __SCHED_UTIL_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <time.h>
#include <stringlist.h>

#define SCHED_KW_DEFAULT_ITEM "*"

char * sched_util_alloc_line(FILE *, bool *);
void   sched_util_parse_line(const char * , int * , char *** , int , bool *);
void   sched_util_parse_file(const char *, int *, char ***);

char * sched_util_alloc_next_entry(FILE *, bool *, bool *);
char * sched_util_alloc_slash_terminated_line(FILE * stream);

void   sched_util_fprintf_int(bool ,          int , int , FILE *);
void   sched_util_fprintf_dbl(bool , double , int , int , FILE *);
double sched_util_atof(const char *);
int    sched_util_atoi(const char *);
void   sched_util_fprintf_qst(bool , const char * , int , FILE *);
void   sched_util_fprintf_tokenlist(int num_token , const char ** token_list , const bool * def);
void   sched_util_skip_trailing_tokens( const stringlist_type * tokens , int * __token_index );
void   sched_util_skip_newline( const stringlist_type * tokens , int * __token_index );
stringlist_type * sched_util_alloc_line_tokens( const stringlist_type * tokens , bool untyped , int num_tokens , int * __token_index);
void              sched_util_init_default(const stringlist_type * line_tokens , bool * def);

#ifdef __cplusplus
}
#endif
#endif
