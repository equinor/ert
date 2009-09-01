#ifndef __BASIC_QUEUE_DRIVER_H__
#define __BASIC_QUEUE_DRIVER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <util.h>

typedef enum {job_queue_null          =  0 ,   /* For a queue node which has been allocated - but not "added" with a job_queue_add_job() call. */
	      job_queue_waiting       =  1 ,   /* A node which is waiting in the internal queue. */
	      job_queue_pending       =  2 ,   /* A node which is pending - a status returned by the external system. I.e LSF */
	      job_queue_running       =  3 ,   /* The job is running */
	      job_queue_done          =  4 ,   /* The job is done - but we have not yet checked if the target file is produced */
	      job_queue_exit          =  5 ,   /* The job has exited - check attempts to determine if we retry or go to complete_fail */
	      job_queue_run_OK        =  6 ,   /* The job has completed - and all checks performed by the queue layer indicate success. */
	      job_queue_run_FAIL      =  7 ,   /* The job has completed - but the queue system has detected that it has failed. */
              job_queue_all_OK        =  8 ,   /* The job has loaded OK - observe that it is the calling scope which will set the status to this. */
	      job_queue_restart       = 10 ,   /* The job is scheduled for a restart. */
	      job_queue_max_state     = 11 } job_status_type;


typedef struct basic_queue_driver_struct basic_queue_driver_type;
typedef struct basic_queue_job_struct    basic_queue_job_type;

typedef basic_queue_job_type * (submit_job_ftype)  	    (void * , int , const char * , const char * , const char * , const void *);
typedef void                   (abort_job_ftype)   	    (void * , basic_queue_job_type * );
typedef job_status_type        (get_status_ftype)  	    (void * , basic_queue_job_type * );
typedef void                   (free_job_ftype)    	    (void * , basic_queue_job_type * );
typedef void                   (free_queue_driver_ftype)    (void *);
typedef void                   (display_info_ftype)         (void * , basic_queue_job_type * );


struct basic_queue_job_struct {
  int __id;
};


#define QUEUE_DRIVER_FUNCTIONS                      \
submit_job_ftype  	   * submit;        	    \
free_job_ftype    	   * free_job;      	    \
abort_job_ftype   	   * abort_f;       	    \
get_status_ftype  	   * get_status;    	    \
free_queue_driver_ftype    * free_driver;   	    \
display_info_ftype         * display_info;


#define BASIC_QUEUE_DRIVER_FIELDS           	    \
submit_job_ftype  	   * submit;        	    \
free_job_ftype    	   * free_job;      	    \
abort_job_ftype   	   * abort_f;       	    \
get_status_ftype  	   * get_status;    	    \
free_queue_driver_ftype    * free_driver;   	    \
display_info_ftype         * display_info;          


struct basic_queue_driver_struct {
  UTIL_TYPE_ID_DECLARATION
  BASIC_QUEUE_DRIVER_FIELDS
};






#ifdef __cplusplus
}
#endif
#endif
