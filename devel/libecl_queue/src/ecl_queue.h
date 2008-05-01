#ifndef __ECL_QUEUE_H__
#define __ECL_QUEUE_H__
#include <basic_queue_driver.h>
#include <path_fmt.h>

typedef struct ecl_queue_struct ecl_queue_type;
void                ecl_queue_finalize(ecl_queue_type * queue);
ecl_queue_type   *  ecl_queue_alloc(int  , int , int  , const char * , const char * , const char * , const char *  , const char *  , const path_fmt_type * , const path_fmt_type * , const path_fmt_type * , void * );
void                ecl_queue_free(ecl_queue_type *);
void                ecl_queue_add_job(ecl_queue_type * , int , int);
void                ecl_queue_run_jobs(ecl_queue_type * , int );
void *              ecl_queue_run_jobs__(void * );
ecl_job_status_type ecl_queue_export_job_status(ecl_queue_type * , int );
#endif
