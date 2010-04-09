#ifndef __JOB_QUEUE_H__
#define __JOB_QUEUE_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <basic_queue_driver.h>
#include <path_fmt.h>

typedef struct job_queue_struct job_queue_type;

job_driver_type     job_queue_get_driver_type( const job_queue_type * queue );
void                job_queue_set_driver(job_queue_type * queue , basic_queue_driver_type * driver);
void                job_queue_set_size( job_queue_type * job_queue , int size );
void                job_queue_finalize(job_queue_type * queue);
void                job_queue_set_runpath_fmt(job_queue_type *  , const path_fmt_type * );
job_queue_type   *  job_queue_alloc(int  , int , int  , const char * );
void                job_queue_free(job_queue_type *);
void                job_queue_add_job(job_queue_type * , const char * , const char * , int , const void *);
void                job_queue_run_jobs(job_queue_type * , int , bool verbose);
void *              job_queue_run_jobs__(void * );
job_status_type     job_queue_export_job_status(job_queue_type * , int );
void                job_queue_set_load_OK(job_queue_type * queue , int external_id);
void                job_queue_set_all_fail(job_queue_type * queue , int external_id);
void                job_queue_set_external_restart(job_queue_type * queue , int external_id);
void                job_queue_set_external_fail(job_queue_type * queue , int external_id);
void                job_queue_set_external_load(job_queue_type * queue , int external_id);
const char        * job_queue_status_name( job_status_type status );
void                job_queue_set_max_running( job_queue_type * queue , int max_running );
int                 job_queue_inc_max_runnning( job_queue_type * queue, int delta );
int                 job_queue_get_max_running( const job_queue_type * queue );
void                job_queue_export_status_summary( job_queue_type * queue , int * external_status_list);
int                 job_queue_iget_status_summary( const job_queue_type * queue , job_status_type status);
time_t              job_queue_iget_sim_start( job_queue_type * queue, int external_id);

#ifdef __cplusplus
}
#endif
#endif

