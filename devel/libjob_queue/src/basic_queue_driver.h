#ifndef __BASIC_QUEUE_DRIVER_H__
#define __BASIC_QUEUE_DRIVER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <util.h>

typedef enum { NULL_DRIVER  = 0,
               LSF_DRIVER   = 1,
               LOCAL_DRIVER = 2,
               RSH_DRIVER   = 3} job_driver_type;


typedef enum { JOB_QUEUE_NOT_ACTIVE    =    1 ,   /* This value is used in external query routines - for jobs which are (currently) not active. */
               JOB_QUEUE_LOADING       =    2 ,   /* This value is used by external routines. Not used in the libjob_queue implementation. */
	       JOB_QUEUE_WAITING       =    4 ,   /* A node which is waiting in the internal queue. */
               JOB_QUEUE_SUBMITTED     =    8 ,   /* Internal status: It has has been submitted - the next status update will (should) place it as pending or running. */
	       JOB_QUEUE_PENDING       =   16 ,   /* A node which is pending - a status returned by the external system. I.e LSF */
	       JOB_QUEUE_RUNNING       =   32 ,   /* The job is running */
	       JOB_QUEUE_DONE          =   64 ,   /* The job is done - but we have not yet checked if the target file is produced */
	       JOB_QUEUE_EXIT          =  128 ,   /* The job has exited - check attempts to determine if we retry or go to complete_fail   */
	       JOB_QUEUE_RUN_OK        =  256 ,   /* The job has completed - and all checks performed by the queue layer indicate success. */
	       JOB_QUEUE_RUN_FAIL      =  512 ,   /* The job has completed - but the queue system has detected that it has failed.         */
               JOB_QUEUE_ALL_OK        = 1024 ,   /* The job has loaded OK - observe that it is the calling scope which will set the status to this. */
               JOB_QUEUE_ALL_FAIL      = 2048 ,   /* The job has failed completely - the calling scope must set this status. */
               JOB_QUEUE_USER_KILLED   = 4096 ,   /* The job has been killed by the user - can restart. */
               JOB_QUEUE_USER_EXIT     = 8192 }   /* The whole job_queue has been exited by the user - the job can NOT be restarted. */
               job_status_type;
#define JOB_QUEUE_MAX_STATE 14


/*
  All jobs which are in the status set defined by
  JOB_QUEUE_CAN_RESTART can be restarted based on external
  user-input. It is OK to try to restart a job which is not in this
  state - basically nothing should happen.
*/
#define JOB_QUEUE_CAN_RESTART  (JOB_QUEUE_ALL_FAIL + JOB_QUEUE_USER_KILLED  +  JOB_QUEUE_ALL_OK)


/*
  These are the jobs which can be killed. It is OK to try to kill a
  job which is not in this state, the only thing happening is that the
  function job_queue_kill_simulation() wil return false.
*/
#define JOB_QUEUE_CAN_KILL    (JOB_QUEUE_WAITING + JOB_QUEUE_RUNNING + JOB_QUEUE_PENDING + JOB_QUEUE_SUBMITTED)


/*
  An external thread is watching the queue (enkf_main_wait_loop()),
  and sending instructions to load (and verfiy) the results when the
  queue says that the jobs have completed. This external
  "queue-watcher" will exit when all jobs are in one of the states in
  JOB_QUEUE_CAN_FINALIZE.
*/
#define JOB_QUEUE_CAN_FINALIZE (JOB_QUEUE_NOT_ACTIVE + JOB_QUEUE_USER_EXIT + JOB_QUEUE_ALL_FAIL + JOB_QUEUE_ALL_OK)   



#define JOB_QUEUE_CAN_UPDATE_STATUS (JOB_QUEUE_RUNNING + JOB_QUEUE_PENDING + JOB_QUEUE_SUBMITTED)


typedef struct basic_queue_driver_struct basic_queue_driver_type;

typedef void                 * (submit_job_ftype)  	    (void * , const char * , const char * , const char * , int argc , const char **);
typedef void                   (kill_job_ftype)   	    (void * , void * );
typedef job_status_type        (get_status_ftype)  	    (void * , void * );
typedef void                   (free_job_ftype)    	    (void * );
typedef void                   (free_queue_driver_ftype)    (void *); 
typedef void                   (display_info_ftype)         (void * , void * );


struct basic_queue_job_struct {
  int __id;
};


#define QUEUE_DRIVER_FUNCTIONS                      \
submit_job_ftype  	   * submit;        	    \
free_job_ftype    	   * free_job;      	    \
kill_job_ftype   	   * kill_job;              \
get_status_ftype  	   * get_status;    	    \
free_queue_driver_ftype    * free_driver;   	    \
display_info_ftype         * display_info;          \
job_driver_type              driver_type;


struct basic_queue_driver_struct {
  UTIL_TYPE_ID_DECLARATION
  QUEUE_DRIVER_FUNCTIONS
};




#ifdef __cplusplus
}
#endif
#endif
