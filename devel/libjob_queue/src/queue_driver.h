#ifndef __QUEUE_DRIVER_H__
#define __QUEUE_DRIVER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <util.h>
#include <hash.h>

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


typedef struct queue_driver_struct queue_driver_type;

typedef void                 * (submit_job_ftype)           (void * , const char * , const char * , const char * , int argc , const char **);
typedef void                   (kill_job_ftype)             (void * , void * );
typedef job_status_type        (get_status_ftype)           (void * , void * );
typedef void                   (free_job_ftype)             (void * );
typedef void                   (free_queue_driver_ftype)    (void *); 
typedef void                   (set_option_ftype)           (void * , int , const void * );
typedef const void *           (get_option_ftype)           (const void * , int );
typedef bool                   (has_option_ftype)           (const void * , int );


queue_driver_type * queue_driver_alloc_RSH( const char * rsh_cmd , const hash_type * rsh_hostlist);
queue_driver_type * queue_driver_alloc_LSF(const char * queue_name , const char * resource_request , const char * remote_lsf_server , int num_cpu);
queue_driver_type * queue_driver_alloc_local( );

void *              queue_driver_submit_job( queue_driver_type * driver, const char * run_cmd , const char * run_path , const char * job_name , int argc , const char ** argv);
void                queue_driver_free_job( queue_driver_type * driver , void * job_data );
void                queue_driver_kill_job( queue_driver_type * driver , void * job_data );
job_status_type     queue_driver_get_status( queue_driver_type * driver , void * job_data);

void                queue_driver_set_max_running( queue_driver_type * driver , int max_running);
int                 queue_driver_get_max_running( const queue_driver_type * driver );
const char        * queue_driver_get_name( const queue_driver_type * driver );

void                queue_driver_set_option( queue_driver_type * driver , int option_id , const void * value);
void                queue_driver_set_int_option( queue_driver_type * driver , int option_id , int int_value);
const        void * queue_driver_get_option( queue_driver_type * driver , int option_id );

void                queue_driver_free( queue_driver_type * driver );

#ifdef __cplusplus
}
#endif
#endif
