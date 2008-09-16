#ifndef __SCHED_FILE_H__
#define __SCHED_FILE_H__
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <list.h>

typedef struct sched_file_struct sched_file_type;

//int               sched_file_get_volume(const sched_file_type *);
list_type       * sched_file_get_kw_list(const sched_file_type * );
time_t            sched_file_get_start_date(const sched_file_type * );
//void 		  sched_file_init_conn_factor(sched_file_type * , const char *  , bool , const int * );
//void 		  sched_file_set_conn_factor(sched_file_type *  , const float * , const float * , const int * );
sched_file_type * sched_file_alloc(time_t);
void              sched_file_fprintf(const sched_file_type * , int , time_t , double , const char * );
sched_file_type * sched_file_fread_alloc(FILE *, int , time_t , double );
void              sched_file_fwrite(const sched_file_type * , FILE *stream );
void              sched_file_parse(sched_file_type * , const char * );
void              sched_file_free(sched_file_type *);
void              sched_file_fprintf_rates(const sched_file_type * , const char * , const char *);
void              sched_file_fprintf_days_dat(const sched_file_type * , const char *);
int 		  sched_file_time_t_to_report_step(const sched_file_type *  , time_t );
int 		  sched_file_time_t_to_report_step__(const sched_file_type *  , time_t , int *);
//int 		  sched_file_int3_to_report_step(const sched_file_type * , int , int , int , int * );
int 		  sched_file_DATES_to_report_step(const sched_file_type * , const char * , int *);
time_t            sched_file_DATES_to_time_t(const sched_file_type *  , const char * );
bool              sched_file_has_well(const sched_file_type * , const char * );
int               sched_file_count_report_steps(const sched_file_type * );
time_t            sched_file_report_step_to_time_t(const sched_file_type *  , int );
#endif
