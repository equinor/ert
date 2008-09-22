#ifndef __JOB_QUEUE_H__
#define __JOB_QUEUE_H__
#include <basic_queue_driver.h>
#include <path_fmt.h>

typedef struct job_queue_struct job_queue_type;
void                job_queue_finalize(job_queue_type * queue);
void                job_queue_set_runpath_fmt(job_queue_type *  , const path_fmt_type * );
job_queue_type   *  job_queue_alloc(int  , int , int  , const char * , void * );
void                job_queue_free(job_queue_type *);
void                job_queue_add_job(job_queue_type * , const char * , const char * , int , int);
void                job_queue_run_jobs(job_queue_type * , int );
void *              job_queue_run_jobs__(void * );
job_status_type     job_queue_export_job_status(job_queue_type * , int );
#endif
