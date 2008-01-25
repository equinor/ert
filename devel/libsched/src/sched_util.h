#ifndef __SCHED_UTIL_H__
#define __SCHED_UTIL_H__
#include <stdbool.h>
#include <time.h>

void sched_util_parse_line(const char * , int * , char *** , int , bool *);
void sched_util_free_token_list(int , char **);
void sched_util_parse_file(const char *, int *, char ***);

void   sched_util_fprintf_int(bool ,          int , int , FILE *);
void   sched_util_fprintf_dbl(bool , double , int , int , FILE *);
double sched_util_atof(const char *);
int    sched_util_atoi(const char *);
void   sched_util_fprintf_qst(bool , const char * , int , FILE *);
void sched_util_fprintf_days_line(int , time_t , time_t , FILE *);

#endif
