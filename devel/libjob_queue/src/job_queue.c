/*
   Copyright (C) 2011  Statoil ASA, Norway.

   The file 'job_queue.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#define  _GNU_SOURCE   /* Must define this to get access to pthread_rwlock_t */
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#include <ert/util/msg.h>
#include <ert/util/util.h>
#include <ert/util/thread_pool.h>
#include <ert/util/arg_pack.h>

#include <ert/job_queue/job_queue.h>
#include <ert/job_queue/job_node.h>
#include <ert/job_queue/job_list.h>
#include <ert/job_queue/queue_driver.h>



#define JOB_QUEUE_START_SIZE 16

/**
   The running of external jobs is handled thruogh an abstract
   job_queue implemented in this file; the job_queue then contains a
   'driver' which actually runs the job. All drivers must support the
   following functions

     submit: This will submit a job, and return a pointer to a
             newly allocated queue_job instance.

     clean:  This will clear up all resources used by the job.

     abort:  This will stop the job, and then call clean.

     status: This will get the status of the job.


   When calling the various driver functions the queue layer needs to
   dereference the driver structures, i.e. to get access to the
   driver->submit_jobs function. This is currently (rather clumsily??
   implemented like this):

        When implementing a driver the driver struct MUST start like
        this:

        struct some_driver {
            UTIL_TYPE_ID_DECLARATION
            QUEUE_DRIVER_FUNCTIONS
            ....
            ....
        }

        The function allocating a driver instance will just return a
        (void *) however in the queue layer the driver is stored as a
        queue_driver_type instance which is a struct like this:

        struct queue_driver_struct {
            UTIL_TYPE_ID_DECLARATION
            QUEUE_DRIVER_FIELDS
        }

        I.e. it only contains the pointers common to all the driver
        implementations. When calling a driver function the spesific
        driver will cast to it's datatype.

   Observe that this library also contains the files ext_joblist and
   ext_job, those files implement a particular way of dispatching
   external jobs in a series; AFTER THEY HAVE BEEN SUBMITTED. So seen
   from the this scope those files do not provide any particluar
   functionality; there is no compile-time dependencies either.
*/


/*
  Threads and such.
  =================

  The job_queue is executed with mulitple threads, and the potential for
  thread-related fuckups is immense. There are essentially two different scopes
  which acces the internal state of the queue concurrently:

    1. The function job_queue_run_jobs() is the main function administrating the
       queue, this includes starting and stopping jobs, and collecting the
       status of the various jobs. The thread running this function is the
       default 'owner' of the information in the job_queue instance.

    2. External scope can:

       o Query the status of the queue / individual jobs.        [Read access]

       o Issue commands to make the queue resubmit/kill/wait/... [Write access]

  Observe that there can be maaany concurrent invokations of the second
  type. Data structures which can change must obviously be protected with
  read/write locking, however scalars are NOT protected, i.e the two code blocks:


     ...
     state = new_value;
     ...

  and

     ...
     return state;

  can run concurrently. In principle we might risk that the return value from
  "return state;" is inconsistent, i.e. different from both new_value and the
  value state had prior to the statement "state = new_value;" - however all
  tests should be explicit so that such an inconsistency is actually OK.

*/


/*
  Some words about status
  =======================

  The status of a particular job is given by the job_status field of
  the job_queue_node_type, the possible values are given by the enum
  job_status_type, defined in queue_driver.h.

  To actually __GET__ the status of a job we use the driver->status()
  function which will invoke a driver specific function and return the
  new status.

    1. The driver->status() function is invoked by the
       job_queue_update_status() function. This should be invoked by
       the same thread as is running the main queue management in
       job_queue_run_jobs().


    2. The actual change of status is handled by the function
       job_queue_change_node_status(); arbitrary assignments of the
       type job->status = new_status is STRICTLY ILLEGAL.


    3. When external functions query about the status of a particular
       job they get the status value currently stored (i.e. cached) in
       the job_node; external scope can NOT initiate a
       driver->status() function call.

       This might result in external scope getting a outdated status -
       live with it.


    4. The name 'status' indicates that this is read-only property;
       that is actually not the case. In the main manager function
       job_queue_run_jobs() action is taken based on the value of the
       status field, and to initiate certain action on jobs the queue
       system (and also external scope) can explicitly set the status
       of a job (by using the job_queue_change_node_status() function).

       The most promiment example of this is when we want to run a
       certain job again, that is achieved with:

           job_queue_node_change_status( queue , node , JOB_QUEUE_WAITING );

       When the queue manager subsequently finds the job with status
       'JOB_QUEUE_WAITING' it will (re)submit this job.
*/



/*
  Communicating success/failure between the job_script and the job_queue:
  =======================================================================

  The system for communicatin success/failure between the queue system
  (i.e. this file) and the job script is quite elaborate. There are
  essentially three problems which make this complicated:

   1. The exit status of the jobs is NOT reliably captured - the job
      might very well fail without us detecing it with the exit
      status.

   2. Syncronizing of disks can be quite slow, so altough a job has
      completede successfully the files we expect to find might not
      present.

   3. There is layer upon layer here - this file scope (i.e. the
      internal queue_system) spawns external jobs in the form of a job
      script. This script again spawns a series of real external jobs
      like e.g. ECLIPSE and RMS. The job_script does not reliably
      capture the exit status of the external programs.


  The approach to this is as follows:

   1. If the job (i.e. the job script) finishes with a failure status
      we communicate the failure back to the calling scope with no
      more ado.

   2. When a job has finished (seemingly OK) we try hard to determine
      whether the job has failed or not. This is based on the
      following tests:

      a) If the job has produced an EXIT file it has failed.

      b) If the job has produced an OK file it has succeeded.

      c) If neither EXIT nor OK files have been produced we spin for a
         while waiting for one of the files, if none turn up we will
         eventually mark the job as failed.

*/



typedef enum {SUBMIT_OK           = 0 ,
              SUBMIT_JOB_FAIL     = 1 , /* Typically no more attempts. */
              SUBMIT_DRIVER_FAIL  = 2 , /* The driver would not take the job - for whatever reason?? */
              SUBMIT_QUEUE_CLOSED = 3 } /* The queue is currently not accepting more jobs - either (temporarilty)
                                           because of pause or it is going down. */   submit_status_type;



/**
   This struct holds the job_queue information about one job. Observe
   the following:

    1. This struct is purely static - i.e. it is invisible outside of
       this file-scope.

    2. Typically the driver would like to store some additional
       information, i.e. the PID of the running process for the local
       driver; that is stored in a (driver specific) struct under the
       field job_data.

    3. If the driver detects that a job has failed it leaves an EXIT
       file, the exit status is (currently) not reliably transferred
       back to to the job_queue layer.

*/


static const int status_index[] = {  JOB_QUEUE_NOT_ACTIVE ,  // Initial, allocated job state, job not added                                - controlled by job_queue
                                     JOB_QUEUE_WAITING    ,  // The job is ready to be started                                             - controlled by job_queue
                                     JOB_QUEUE_SUBMITTED  ,  // Job is submitted to driver - temporary state                               - controlled by job_queue
                                     JOB_QUEUE_PENDING    ,  // Job is pending, before actual execution                                    - controlled by queue_driver
                                     JOB_QUEUE_RUNNING    ,  // Job is executing                                                           - controlled by queue_driver
                                     JOB_QUEUE_DONE       ,  // Job is done (sucessful or not), temporary state                            - controlled/returned by by queue_driver
                                     JOB_QUEUE_EXIT       ,  // Job is done, with exit status != 0, temporary state                        - controlled/returned by by queue_driver
                                     JOB_QUEUE_USER_EXIT  ,  // User / queue system has requested killing of job                           - controlled by job_queue / external scope
                                     JOB_QUEUE_USER_KILLED,  // Job has been killed, due to JOB_QUEUE_USER_EXIT, FINAL STATE               - controlled by job_queue
                                     JOB_QUEUE_SUCCESS    ,  // All good, comes after JOB_QUEUE_DONE, with additional checks, FINAL STATE  - controlled by job_queue
                                     JOB_QUEUE_RUNNING_CALLBACK, // Temporary state, while running requested callbacks after an ended job  - controlled by job_queue
                                     JOB_QUEUE_FAILED };     // Job has failed, no more retries, FINAL STATE

static const char* status_name[] = { "JOB_QUEUE_NOT_ACTIVE" ,
                                     "JOB_QUEUE_WAITING"    ,
                                     "JOB_QUEUE_SUBMITTED"  ,
                                     "JOB_QUEUE_PENDING"    ,
                                     "JOB_QUEUE_RUNNING"    ,
                                     "JOB_QUEUE_DONE"       ,
                                     "JOB_QUEUE_EXIT"       ,
                                     "JOB_QUEUE_USER_KILLED" ,
                                     "JOB_QUEUE_USER_EXIT"   ,
                                     "JOB_QUEUE_SUCCESS"    ,
                                     "JOB_QUEUE_RUNNING_CALLBACK",
                                     "JOB_QUEUE_FAILED" };



/*****************************************************************/


/**

   This is the struct for a whole queue. Observe the following:

    1. The number of elements is specified at the allocation time, and
       all nodes are allocated then; i.e. when xx_insert_job() is called
       from external scope a new node is not actaully created
       internally, it is just an existing node which is initialized.

    2. The queue can start running before all jobs are added.

*/

struct job_queue_struct {
  //int                        active_size;                       /* The current number of job slots in the queue. */
  //int                        alloc_size;                        /* The current allocated size of jobs array. */
  //job_queue_node_type     ** jobs;                              /* A vector of job nodes .*/
  job_list_type            * job_list;
  int                        max_submit;                        /* The maximum number of submit attempts for one job. */
  char                     * exit_file;                         /* The queue will look for the occurence of this file to detect a failure. */
  char                     * ok_file;                           /* The queue will look for this file to verify that the job was OK - can be NULL - in which case it is ignored. */
  queue_driver_type       * driver;                             /* A pointer to a driver instance (LSF|LOCAL|RSH) which actually 'does it'. */
  int                        status_list[JOB_QUEUE_MAX_STATE];  /* The number of jobs in the different states. */
  int                        old_status_list[JOB_QUEUE_MAX_STATE]; /* Should the display be updated ?? */
  bool                       open;                              /* True if the queue has been reset and is ready for use, false if the queue has been used and not reset */
  bool                       user_exit;                         /* If there comes an external signal to abondond the whole thing user_exit will be set to true, and things start to dwindle down. */
  bool                       running;
  bool                       pause_on;
  bool                       submit_complete;
  bool                       grow;                              /* The function adding new jobs is requesting the job_queue function to grow the jobs array. */
  int                        max_ok_wait_time;                  /* Seconds to wait for an OK file - when the job itself has said all OK. */
  int                        max_duration;                      /* Maximum allowed time for a job to run, 0 = unlimited */
  time_t                     stop_time;                         /* A job is only allowed to run until this time. 0 = no time set, ignore stop_time */
  unsigned long              usleep_time;                       /* The sleep time before checking for updates. */
  pthread_mutex_t            status_mutex;                      /* This mutex ensure that the status-change code is only run by one thread. */
  pthread_mutex_t            run_mutex;                         /* This mutex is used to ensure that ONLY one thread is executing the job_queue_run_jobs(). */
  pthread_mutex_t            submit_mutex;                      /* This mutex ensures that ONLY one theread is adding jobs to the queue. */
  pthread_rwlock_t           queue_lock;                        /* This a rwlock around the jobs datastructure. */
  thread_pool_type         * work_pool;
};

/*****************************************************************/

job_queue_node_type * job_queue_iget_node(job_queue_type * queue , int job_index) {
  job_queue_node_type * node = NULL;

  pthread_rwlock_rdlock( &queue->queue_lock );
  node = job_list_iget_job( queue->job_list , job_index );
  pthread_rwlock_unlock( &queue->queue_lock );

  return node;
}



static int STATUS_INDEX( job_status_type status ) {
  int index = 0;

  while (true) {
    if (status_index[index] == status)
      return index;

    index++;
    if (index == JOB_QUEUE_MAX_STATE)
      util_abort("%s: failed to get index from status:%d \n",__func__ , status);
  }
}

/*
  int status_index = 0;
  int status = input_status;
  while ( (status != 1) && (status_index < JOB_QUEUE_MAX_STATE)) {
  status >>= 1;
  status_index++;
  }
  if (status != 1)
  util_abort("%s: failed to get index from status:%d \n",__func__ , status);
  return status_index;
}
*/



/*****************************************************************/





/*****************************************************************/

static bool job_queue_change_node_status(job_queue_type *  , job_queue_node_type *  , job_status_type );







/**
   This function WILL be called by several threads concurrently; both
   directly from the thread running the job_queue_run_jobs() function,
   and indirectly thorugh exported functions like:

      job_queue_set_external_restart();
      job_queue_set_external_fail();
      ...

   It is therefor essential that only one thread is running this code
   at time.
*/

static bool job_queue_change_node_status(job_queue_type * queue , job_queue_node_type * node , job_status_type new_status) {
  bool status_change = false;
  pthread_mutex_lock( &queue->status_mutex );
  {
    job_status_type old_status = job_queue_node_get_status( node );

    if (new_status != old_status) {
      job_queue_node_update_status( node , new_status );

      queue->status_list[ STATUS_INDEX(old_status) ]--;
      queue->status_list[ STATUS_INDEX(new_status) ]++;

      status_change = true;
    }
  }
  pthread_mutex_unlock( &queue->status_mutex );
  return status_change;
}



/*
   This frees the storage allocated by the driver - the storage
   allocated by the queue layer is retained.

   In the case of jobs which are first marked as successfull by the
   queue layer, and then subsequently set to status EXIT by the
   DONE_callback this function will be called twice; i.e. we must
   protect against a double free.
*/

static void job_queue_free_job_driver_data(job_queue_type * queue , job_queue_node_type * node) {
  job_queue_node_get_wrlock(node);
  job_queue_node_free_driver_data( node , queue->driver );
  job_queue_node_unlock(node);
}



/**
   Observe that this function should only query the driver for state
   change when the job is currently in one of the states:

     JOB_QUEUE_WAITING || JOB_QUEUE_PENDING || JOB_QUEUE_RUNNING

   The other state transitions are handled by the job_queue itself,
   without consulting the driver functions.
*/

/*
   Will return true if the status has changed since the last time.
*/

static bool job_queue_update_status(job_queue_type * queue ) {
  bool update = false;
  queue_driver_type *driver  = queue->driver;
  int ijob;


  for (ijob = 0; ijob < job_list_get_size( queue->job_list ); ijob++) {
    job_queue_node_type * node = job_list_iget_job( queue->job_list , ijob );

    job_queue_node_get_rdlock( node );
    {
      void * node_data = job_queue_node_get_data( node );
      if (node_data) {
        job_status_type current_status = job_queue_node_get_status(node);
        if (current_status & JOB_QUEUE_CAN_UPDATE_STATUS) {
          job_status_type new_status = queue_driver_get_status( driver , node_data);
          job_queue_change_node_status(queue , node , new_status);
        }
      }
    }
    job_queue_node_unlock( node );

  }

  /* Has the net status changed? */
  {
    int istat;
    for (istat = 0; istat  < JOB_QUEUE_MAX_STATE; istat++) {
      if (queue->old_status_list[istat] != queue->status_list[istat])
        update = true;
      queue->old_status_list[istat] = queue->status_list[istat];
    }
  }
  return update;
}



static submit_status_type job_queue_submit_job(job_queue_type * queue , int queue_index) {
  submit_status_type submit_status;
  if (queue->user_exit || queue->pause_on)
    submit_status = SUBMIT_QUEUE_CLOSED;   /* The queue is currently not accepting more jobs. */
  else {
    {
      job_queue_node_type * node = job_list_iget_job( queue->job_list , queue_index );
      void * job_data = queue_driver_submit_job( queue->driver  ,
                                                 job_queue_node_get_cmd( node ),
                                                 job_queue_node_get_num_cpu( node ),
                                                 job_queue_node_get_run_path( node ),
                                                 job_queue_node_get_name( node ),
                                                 job_queue_node_get_argc( node ),
                                                 job_queue_node_get_argv( node ) );

      if (job_data != NULL) {
        job_queue_node_get_wrlock( node );
        {
          job_queue_node_update_data( node , job_data );
          job_queue_node_inc_submit_attempt( node );
          job_queue_change_node_status(queue , node , JOB_QUEUE_SUBMITTED );

          /*
             The status JOB_QUEUE_SUBMITTED is internal, and not
             exported anywhere. The job_queue_update_status() will
             update this to PENDING or RUNNING at the next call. The
             important difference between SUBMITTED and WAITING is
             that SUBMITTED have job_data != NULL and the
             job_queue_node free function must be called on it.
          */

          submit_status = SUBMIT_OK;
        }
        job_queue_node_unlock( node );
      } else
        /*
          In this case the status of the job itself will be
          unmodified; i.e. it will still be WAITING, and a new attempt
          to submit it will be performed in the next round.
        */
        submit_status = SUBMIT_DRIVER_FAIL;
    }
  }
  return submit_status;
}






const char * job_queue_iget_run_path( job_queue_type * queue , int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
  return job_queue_node_get_run_path( node );
}


const char * job_queue_iget_failed_job( job_queue_type * queue , int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
  return job_queue_node_get_failed_job( node );
}


const char * job_queue_iget_error_reason( job_queue_type * queue , int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
  return job_queue_node_get_error_reason( node );
}


const char * job_queue_iget_stderr_capture(  job_queue_type * queue , int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
    return job_queue_node_get_stderr_capture( node );
}


const char * job_queue_iget_stderr_file( job_queue_type * queue , int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
    return job_queue_node_get_stderr_file( node );
}




job_status_type job_queue_iget_job_status( job_queue_type * queue , int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
  return job_queue_node_get_status( node );
}



/**
   Will return the number of jobs with status @status.

      #include <queue_driver.h>

      printf("Running jobs...: %03d \n", job_queue_iget_status_summary( queue , JOB_QUEUE_RUNNING ));
      printf("Waiting jobs:..: %03d \n", job_queue_iget_status_summary( queue , JOB_QUEUE_WAITING ));

   Observe that if this function is called repeatedly the status might change between
   calls, with the consequence that the total number of jobs does not add up
   properly. The handles itself autonomously so as long as the return value from this
   function is only used for information purposes this does not matter. Alternatively
   the function job_queue_export_status_summary(), which does proper locking, can be
   used.
*/

int job_queue_iget_status_summary( const job_queue_type * queue , job_status_type status) {
  return queue->status_list[ STATUS_INDEX( status ) ];
}


int job_queue_get_num_running( const job_queue_type * queue) {
  return job_queue_iget_status_summary( queue , JOB_QUEUE_RUNNING );
}

int job_queue_get_num_pending( const job_queue_type * queue) {
  return job_queue_iget_status_summary( queue , JOB_QUEUE_PENDING );
}

int job_queue_get_num_waiting( const job_queue_type * queue) {
  return job_queue_iget_status_summary( queue , JOB_QUEUE_WAITING );
}

int job_queue_get_num_complete( const job_queue_type * queue) {
  return job_queue_iget_status_summary( queue , JOB_QUEUE_SUCCESS );
}

int job_queue_get_num_failed( const job_queue_type * queue) {
  return job_queue_iget_status_summary( queue , JOB_QUEUE_FAILED );
}

int job_queue_get_num_killed( const job_queue_type * queue) {
  return job_queue_iget_status_summary( queue , JOB_QUEUE_USER_KILLED );
}

int job_queue_get_active_size( const job_queue_type * queue ) {
  return job_list_get_size( queue->job_list );
}

void job_queue_set_max_job_duration(job_queue_type * queue, int max_duration_seconds) {
  queue->max_duration = max_duration_seconds;
}

int job_queue_get_max_job_duration(const job_queue_type * queue) {
  return queue->max_duration;
}

void job_queue_set_job_stop_time(job_queue_type * queue, time_t time) {
  queue->stop_time = time;
}

time_t job_queue_get_job_stop_time(const job_queue_type * queue) {
  return queue->stop_time;
}

void job_queue_set_auto_job_stop_time(job_queue_type * queue) {
  time_t sum_run_time_succeded_jobs = 0;
  int num_succeded_jobs = 0;

  for (int i = 0; i < job_list_get_size( queue->job_list ); i++) {
    if (JOB_QUEUE_SUCCESS == job_queue_iget_job_status(queue,i)) {
      sum_run_time_succeded_jobs += difftime(job_queue_iget_sim_end(queue, i), job_queue_iget_sim_start(queue, i));
      ++num_succeded_jobs;
    }
  }

  if (num_succeded_jobs > 0) {
    time_t avg_run_time_succeded_jobs = sum_run_time_succeded_jobs / num_succeded_jobs;
    time_t stop_time = time(NULL) + (avg_run_time_succeded_jobs * 0.25);
    job_queue_set_job_stop_time(queue, stop_time);
  }
}

/**
   Observe that jobs with status JOB_QUEUE_WAITING can also be killed; for those
   jobs the kill should be interpreted as "Forget about this job for now and set
   the status JOB_QUEUE_USER_KILLED", however it is important that we not call
   the driver->kill() function on it because the job slot will have no data
   (i.e. LSF jobnr), and the driver->kill() function will fail if presented with
   such a job.

   Only jobs which have a status matching "JOB_QUEUE_CAN_KILL" can be
   killed; if the job is not in a killable state the function will do
   nothing. This includes trying to kill a job which is not even
   found.

   Observe that jobs (slots) with status JOB_QUEUE_NOT_ACTIVE can NOT be
   meaningfully killed; that is because these jobs have not yet been submitted
   to the queue system, and there is not yet established a mapping between
   external id and queue_index.
*/
bool job_queue_kill_job_node( job_queue_type * queue , job_queue_node_type * node) {
  bool result = false;

  job_queue_node_get_wrlock( node );
  {
    job_status_type status = job_queue_node_get_status( node );
    if (status & JOB_QUEUE_CAN_KILL) {
      queue_driver_type * driver = queue->driver;
      /*
         Jobs with status JOB_QUEUE_WAITING are killable - in the
         sense that status should be set to JOB_QUEUE_USER_KILLED; but
         they do not have any driver specific job_data, and the
         driver->kill_job() function can NOT be called.
      */
      if (status != JOB_QUEUE_WAITING)
        job_queue_node_driver_kill( node , driver );

      job_queue_change_node_status( queue , node , JOB_QUEUE_USER_KILLED );
      result = true;
    }
  }
  job_queue_node_unlock( node );
  return result;
}

bool job_queue_kill_job( job_queue_type * queue , int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
  return job_queue_kill_job_node(queue, node);
}


/**
   The external scope asks the queue to restart the the job; we reset
   the submit counter to zero. This function should typically be used
   in combination with resampling, however that is the responsability
   of the calling scope.
*/

void job_queue_iset_external_restart(job_queue_type * queue , int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
  job_queue_node_inc_submit_attempt(node);
  job_queue_change_node_status( queue , node , JOB_QUEUE_WAITING );
}


/**
   The queue system has said that the job completed OK, however the
   external scope failed to load all the results and are using this
   function to inform the queue system that the job has indeed
   failed. The queue system will then either retry the job, or switch
   status to JOB_QUEUE_RUN_FAIL.


   This is a bit dangerous beacuse the queue system has said that the
   job was all hunkadory, and freed the driver related resources
   attached to the job; it is therefor essential that the
   JOB_QUEUE_EXIT code explicitly checks the status of the job node's
   driver specific data before dereferencing.
*/

void job_queue_iset_external_fail(job_queue_type * queue , int job_index) {
  job_queue_node_type * node = job_list_iget_job( queue->job_list , job_index );
  job_queue_change_node_status( queue , node , JOB_QUEUE_EXIT);
}



time_t job_queue_iget_sim_start( job_queue_type * queue, int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
  return job_queue_node_get_sim_start( node );
}

time_t job_queue_iget_sim_end( job_queue_type * queue, int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
  return job_queue_node_get_sim_end( node );
}

time_t job_queue_iget_submit_time( job_queue_type * queue, int job_index) {
  job_queue_node_type * node = job_queue_iget_node( queue, job_index );
  return job_queue_node_get_submit_time( node );
}



static void job_queue_update_spinner( int * phase ) {
  const char * spinner = "-\\|/";
  int spinner_length   = strlen( spinner );

  printf("%c\b" , spinner[ (*phase % spinner_length) ]);
  fflush(stdout);
  (*phase) += 1;
}


static void job_queue_print_summary(const job_queue_type *queue, bool status_change ) {
  const char * status_fmt = "Waiting: %3d    Pending: %3d    Running: %3d    Checking/Loading: %3d    Failed: %3d    Complete: %3d   [ ]\b\b";
  int string_length       = 105;

  if (status_change) {
    for (int i=0; i < string_length; i++)
      printf("\b");
    {
      int waiting  = queue->status_list[ STATUS_INDEX(JOB_QUEUE_WAITING) ];
      int pending  = queue->status_list[ STATUS_INDEX(JOB_QUEUE_PENDING) ];

      /*
         EXIT and DONE are included in "xxx_running", because the target
         file has not yet been checked.
      */
      int running  = queue->status_list[ STATUS_INDEX(JOB_QUEUE_RUNNING) ] + queue->status_list[ STATUS_INDEX(JOB_QUEUE_DONE) ] + queue->status_list[ STATUS_INDEX(JOB_QUEUE_EXIT) ];
      int complete = queue->status_list[ STATUS_INDEX(JOB_QUEUE_SUCCESS) ];
      int failed   = queue->status_list[ STATUS_INDEX(JOB_QUEUE_FAILED) ];
      int loading  = queue->status_list[ STATUS_INDEX(JOB_QUEUE_RUNNING_CALLBACK) ];

      printf(status_fmt , waiting , pending , running , loading , failed , complete);
    }
  }
}






static void job_queue_clear_status( job_queue_type * queue ) {
  for (int i=0; i < JOB_QUEUE_MAX_STATE; i++) {
    queue->status_list[i] = 0;
    queue->old_status_list[i] = 0;
  }
}


/**
    This function goes through all the nodes and call finalize on
    them. What about jobs which were NOT in a CAN_KILL state when the
    killing was done, i.e. jobs which are in one of the intermediate
    load like states?? They
*/

void job_queue_reset(job_queue_type * queue) {
  int i;

  /*
    for (i=0; i < job_list_get_size( queue->job_list ); i++) {
    job_queue_node_type * node = job_list_iget_job( queue->job_list , i );
    job_queue_node_finalize( node );
    }
  */
  job_list_reset( queue->job_list );
  job_queue_clear_status( queue );

  /*
      Be ready for the next run
  */
  queue->grow            = false;
  queue->submit_complete = false;
  queue->pause_on        = false;
  queue->user_exit       = false;
  queue->open            = true;
  queue->stop_time       = 0;
}


bool job_queue_is_running( const job_queue_type * queue ) {
  return queue->running;
}


static void job_queue_user_exit__( job_queue_type * queue ) {
  int queue_index;
  for (queue_index = 0; queue_index < job_list_get_size( queue->job_list ); queue_index++) {
    job_queue_node_type * node = job_list_iget_job( queue->job_list , queue_index );
    job_queue_change_node_status( queue , node , JOB_QUEUE_USER_EXIT);
  }
}


static bool job_queue_check_node_status_files( const job_queue_type * job_queue , job_queue_node_type * node) {
  const char * exit_file = job_queue_node_get_exit_file( node );
  if ((exit_file != NULL) && util_file_exists(exit_file))
    return false;                /* It has failed. */
  else {
    const char * ok_file = job_queue_node_get_ok_file( node );
    if (ok_file == NULL)
      return true;               /* If the ok-file has not been set we just return true immediately. */
    else {
      int ok_sleep_time    =  1; /* Time to wait between checks for OK|EXIT file.                         */
      int  total_wait_time =  0;

      while (true) {
        if (util_file_exists( ok_file )) {
          return true;
          break;
        } else {
          if (total_wait_time <  job_queue->max_ok_wait_time) {
            sleep( ok_sleep_time );
            total_wait_time += ok_sleep_time;
          } else {
            /* We have waited long enough - this does not seem to give any OK file. */
            return false;
            break;
          }
        }
      }
    }
  }
}


static void * job_queue_run_DONE_callback( void * arg ) {
  job_queue_type * job_queue;
  job_queue_node_type * node;
  {
    arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
    job_queue = arg_pack_iget_ptr( arg_pack , 0 );
    node = arg_pack_iget_ptr( arg_pack , 1 );
    arg_pack_free( arg_pack );
  }
  job_queue_free_job_driver_data( job_queue , node );
  {
    bool OK = job_queue_check_node_status_files( job_queue , node );

    if (OK)
      OK = job_queue_node_run_DONE_callback( node );

    if (OK)
      job_queue_change_node_status( job_queue , node , JOB_QUEUE_SUCCESS );
    else
      job_queue_change_node_status( job_queue , node , JOB_QUEUE_EXIT );
  }
  return NULL;
}


static void job_queue_handle_DONE( job_queue_type * queue , job_queue_node_type * node) {
  job_queue_change_node_status(queue , node , JOB_QUEUE_RUNNING_CALLBACK );
  {
    arg_pack_type * arg_pack = arg_pack_alloc();
    arg_pack_append_ptr( arg_pack , queue );
    arg_pack_append_ptr( arg_pack , node );  // Should have a private node copy
    thread_pool_add_job( queue->work_pool , job_queue_run_DONE_callback , arg_pack );
  }
}


static void * job_queue_run_EXIT_callback( void * arg ) {
  job_queue_type * job_queue;
  job_queue_node_type * node;
  {
    arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
    job_queue = arg_pack_iget_ptr( arg_pack , 0 );
    node = arg_pack_iget_ptr( arg_pack , 1 );
    arg_pack_free( arg_pack );
  }
  job_queue_free_job_driver_data( job_queue , node );

  if (job_queue_node_get_submit_attempt( node ) < job_queue->max_submit)
    job_queue_change_node_status( job_queue , node , JOB_QUEUE_WAITING );  /* The job will be picked up for antother go. */
  else {
    bool retry = job_queue_node_run_RETRY_callback( node );

    if (retry) {
      /* OK - we have invoked the retry_callback() - and that has returned true;
         giving this job a brand new start. */
      job_queue_node_reset_submit_attempt( node );
      job_queue_change_node_status(job_queue , node , JOB_QUEUE_WAITING);
    } else {
      // It's time to call it a day

      job_queue_node_run_EXIT_callback( node );
      job_queue_change_node_status(job_queue , node , JOB_QUEUE_FAILED);
    }
  }
  return NULL;
}


static void * job_queue_run_USER_EXIT_callback( void * arg ) {
  job_queue_type * job_queue;
  job_queue_node_type * node;
  {
    arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
    job_queue = arg_pack_iget_ptr( arg_pack , 0 );
    node = arg_pack_iget_ptr( arg_pack , 1 );
    arg_pack_free( arg_pack );
  }
  job_queue_free_job_driver_data( job_queue , node );

  // It's time to call it a day
  job_queue_node_run_EXIT_callback( node );
  job_queue_change_node_status(job_queue, node, JOB_QUEUE_USER_KILLED);
  return NULL;
}

static void job_queue_handle_USER_EXIT( job_queue_type * queue , job_queue_node_type * node) {
  // TODO: Right place for this?
  job_queue_kill_job_node(queue, node);
  job_queue_change_node_status(queue , node , JOB_QUEUE_RUNNING_CALLBACK );
  {
    arg_pack_type * arg_pack = arg_pack_alloc();
    arg_pack_append_ptr( arg_pack , queue );
    arg_pack_append_ptr( arg_pack , node );
    thread_pool_add_job( queue->work_pool , job_queue_run_USER_EXIT_callback , arg_pack );
  }
}

static void job_queue_handle_EXIT( job_queue_type * queue , job_queue_node_type * node) {
  job_queue_change_node_status(queue , node , JOB_QUEUE_RUNNING_CALLBACK );
  {
    arg_pack_type * arg_pack = arg_pack_alloc();
    arg_pack_append_ptr( arg_pack , queue );
    arg_pack_append_ptr( arg_pack , node );
    thread_pool_add_job( queue->work_pool , job_queue_run_EXIT_callback , arg_pack );
  }
}


/*****************************************************************/

int job_queue_get_max_running_option(queue_driver_type * driver) {
  char * max_running_string = (char*)queue_driver_get_option(driver, MAX_RUNNING);
  int max_running;
  if (!util_sscanf_int(max_running_string, &max_running)) {
    fprintf(stderr, "%s: Unable to parse option MAX_RUNNING with value %s to an int", __func__, max_running_string);
  }
  return max_running;
}

void job_queue_set_max_running_option(queue_driver_type * driver, int max_running) {
  char * max_running_string = util_alloc_sprintf("%d", max_running);
  queue_driver_set_option(driver, MAX_RUNNING, max_running_string);
  free(max_running_string);
}


/**
   Observe that if the max number of running jobs is decreased,
   nothing will be done to reduce the number of jobs currently
   running; but no more jobs will be submitted until the number of
   running has fallen below the new limit.

   The updated value will also be pushed down to the current driver.

   NOTE: These next three *max_running functions should not be used, rather
   use the set_option feature, with MAX_RUNNING. They are (maybe) used by python
   therefore not removed.
*/
int job_queue_get_max_running( const job_queue_type * queue ) {
  return job_queue_get_max_running_option(queue->driver);
}

void job_queue_set_max_running( job_queue_type * queue , int max_running ) {
  job_queue_set_max_running_option(queue->driver, max_running);
}

/*
  The return value is the new value for max_running.
*/
int job_queue_inc_max_runnning( job_queue_type * queue, int delta ) {
  job_queue_set_max_running( queue , job_queue_get_max_running( queue ) + delta );
  return job_queue_get_max_running( queue );
}

/*****************************************************************/

static void job_queue_check_expired(job_queue_type * queue) {
  if ((job_queue_get_max_job_duration(queue) <= 0) && (job_queue_get_job_stop_time(queue) <= 0))
    return;

  for (int i = 0; i < job_list_get_size( queue->job_list ); i++) {
    job_queue_node_type * node = job_list_iget_job( queue->job_list , i );

    if (job_queue_node_get_status(node) == JOB_QUEUE_RUNNING) {
      time_t now = time(NULL);
      if ( job_queue_get_max_job_duration(queue) > 0) {
        double elapsed = difftime(now, job_queue_node_get_sim_start( node ));
        if (elapsed > job_queue_get_max_job_duration(queue))
          job_queue_change_node_status(queue, node, JOB_QUEUE_USER_EXIT);
      }
      if (job_queue_get_job_stop_time(queue) > 0) {
        if (now >= job_queue_get_job_stop_time(queue))
          job_queue_change_node_status(queue, node, JOB_QUEUE_USER_EXIT);
      }
    }
  }
}

bool job_queue_get_open(const job_queue_type * job_queue) {
  return job_queue->open;
}

void job_queue_check_open(job_queue_type* queue) {
  if (!job_queue_get_open(queue))
    util_abort("%s: queue not open and not ready for use; method job_queue_reset must be called before using the queue - aborting\n", __func__ );
}

/**
   If the total number of jobs is not known in advance the job_queue_run_jobs
   function can be called with @num_total_run == 0. In that case it is paramount
   to call the function job_queue_submit_complete() whan all jobs have been submitted.

   Observe that this function is assumed to have ~exclusive access to
   the jobs array; meaning that:

     1. The jobs array is read without taking a reader lock.

     2. Other functions accessing the jobs array concurrently must
        take a read lock.

     3. This function should be the *only* function modifying
        the jobs array, and that is done *with* the write lock.

*/

void job_queue_run_jobs(job_queue_type * queue , int num_total_run, bool verbose) {
  int trylock = pthread_mutex_trylock( &queue->run_mutex );
  if (trylock != 0)
    util_abort("%s: another thread is already running the queue_manager\n",__func__);
  else {
    /* OK - we have got an exclusive lock to the run_jobs code. */

    //Check if queue is open. Fails hard if not open
    job_queue_check_open(queue);

    const int NUM_WORKER_THREADS = 16;
    queue->running = true;
    queue->work_pool = thread_pool_alloc( NUM_WORKER_THREADS , true );
    {
      bool new_jobs         = false;
      bool cont             = true;
      int  phase = 0;

      do {
        bool local_user_exit = false;
        /*****************************************************************/
        if (queue->user_exit)  {/* An external thread has called the job_queue_user_exit() function, and we should kill
                                   all jobs, do some clearing up and go home. Observe that we will go through the
                                   queue handling codeblock below ONE LAST TIME before exiting. */
          job_queue_user_exit__( queue );
          local_user_exit = true;
        }

        job_queue_check_expired(queue);

        /*****************************************************************/
        {
          bool update_status = job_queue_update_status( queue );
          if (verbose) {
            if (update_status || new_jobs)
              job_queue_print_summary(queue , update_status );
            job_queue_update_spinner( &phase );
          }


          {
            int num_complete = queue->status_list[ STATUS_INDEX(JOB_QUEUE_SUCCESS)   ] +
                               queue->status_list[ STATUS_INDEX(JOB_QUEUE_FAILED)    ] +
                               queue->status_list[ STATUS_INDEX(JOB_QUEUE_USER_KILLED) ];

            if ((num_total_run > 0) && (num_total_run == num_complete))
              /* The number of jobs completed is equal to the number
                 of jobs we have said we want to run; so we are finished.
              */
              cont = false;
            else {
              if (num_total_run == 0) {
                /* We have not informed about how many jobs we will
                   run. To check if we are complete we perform the two
                   tests:

                     1. All the jobs which have been added with
                        job_queue_add_job() have completed.

                     2. The user has used job_queue_complete_submit()
                        to signal that no more jobs will be forthcoming.
                */
                if ((num_complete == job_list_get_size( queue->job_list )) && queue->submit_complete)
                  cont = false;
              }
            }
          }

          if (cont) {
            /* Submitting new jobs */
            int max_submit     = 5; /* This is the maximum number of jobs submitted in one while() { ... } below.
                                       Only to ensure that the waiting time before a status update is not too long. */
            int total_active   = queue->status_list[ STATUS_INDEX(JOB_QUEUE_PENDING) ] + queue->status_list[ STATUS_INDEX(JOB_QUEUE_RUNNING) ];
            int num_submit_new;

            {
              int max_running = job_queue_get_max_running( queue );
              if (max_running > 0)
                num_submit_new = util_int_min( max_submit ,  max_running - total_active );
              else
                /*
                   If max_running == 0 that should be interpreted as no limit; i.e. the queue layer will
                   attempt to send an unlimited number of jobs to the driver - the driver can reject the jobs.
                */
                num_submit_new = util_int_min( max_submit , queue->status_list[ STATUS_INDEX( JOB_QUEUE_WAITING )]);
            }

            new_jobs = false;
            if (queue->status_list[ STATUS_INDEX(JOB_QUEUE_WAITING) ] > 0)   /* We have waiting jobs at all           */
              if (num_submit_new > 0)                                        /* The queue can allow more running jobs */
                new_jobs = true;

            if (new_jobs) {
              int submit_count = 0;
              int queue_index  = 0;

              while ((queue_index < job_list_get_size( queue->job_list )) && (num_submit_new > 0)) {
                job_queue_node_type * node = job_list_iget_job( queue->job_list , queue_index );
                if (job_queue_node_get_status(node) == JOB_QUEUE_WAITING) {
                  {
                    submit_status_type submit_status = job_queue_submit_job(queue , queue_index);

                    if (submit_status == SUBMIT_OK) {
                      num_submit_new--;
                      submit_count++;
                    } else if ((submit_status == SUBMIT_DRIVER_FAIL) || (submit_status == SUBMIT_QUEUE_CLOSED))
                      break;
                  }
                }
                queue_index++;
              }
            }


            {
              /*
                Checking for complete / exited / overtime jobs
               */
              int queue_index;
              for (queue_index = 0; queue_index < job_list_get_size( queue->job_list ); queue_index++) {
                job_queue_node_type * node = job_list_iget_job( queue->job_list , queue_index );

                switch (job_queue_node_get_status(node)) {
                  case(JOB_QUEUE_DONE):
                    job_queue_handle_DONE(queue, node);
                    break;
                  case(JOB_QUEUE_EXIT):
                    job_queue_handle_EXIT(queue, node);
                    break;
                  case(JOB_QUEUE_USER_EXIT):
                    job_queue_handle_USER_EXIT(queue, node);
                    break;
                  default:
                    break;
                }


              }
            }

            if (local_user_exit)
              cont = false;    /* This is how we signal that we want to get out . */
            else
              if (!new_jobs && cont)
                util_usleep(queue->usleep_time);
          }
        }

      } while ( cont );
      queue->running = false;
    }
    if (verbose)
      printf("\n");
    thread_pool_join( queue->work_pool );
    thread_pool_free( queue->work_pool );
  }

  /*
    Set the queue's "open" flag to false to signal that the queue is
    not ready to be used in a new job_queue_run_jobs or
    job_queue_add_job method call as it has not been reset yet. Not
    resetting the queue here implies that the queue object is still
    available for queries after this method has finished
  */
  queue->open = false;
  pthread_mutex_unlock( &queue->run_mutex );
}



/*
  An external thread sets the user_exit flag to true, then subsequently the
  thread managing the queue will see this, and close down the queue.
*/

void job_queue_user_exit( job_queue_type * queue) {
  queue->user_exit = true;
}



void * job_queue_run_jobs__(void * __arg_pack) {
  arg_pack_type * arg_pack = arg_pack_safe_cast(__arg_pack);
  job_queue_type * queue   = arg_pack_iget_ptr(arg_pack , 0);
  int num_total_run        = arg_pack_iget_int(arg_pack , 1);
  bool verbose             = arg_pack_iget_bool(arg_pack , 2);

  job_queue_run_jobs(queue , num_total_run , verbose);
  arg_pack_free( arg_pack );
  return NULL;
}


void job_queue_start_manager_thread( job_queue_type * job_queue , pthread_t * queue_thread , int job_size , bool verbose) {

  arg_pack_type  * queue_args = arg_pack_alloc(); /* This arg_pack will be freed() in the job_que_run_jobs__() */
  arg_pack_append_ptr(queue_args  , job_queue);
  arg_pack_append_int(queue_args  , job_size);
  arg_pack_append_bool(queue_args , verbose);
  job_queue_reset(job_queue);

  /*
    The running status of the job is set to true here; this is to
    guarantee that if calling scope queries the status of the queue
    before queue_thread has actually started running the queue.
  */
  job_queue->running = true;
  pthread_create( queue_thread , NULL , job_queue_run_jobs__ , queue_args);
}




/**
   The most flexible use scenario is as follows:

     1. The job_queue_run_jobs() is run by one thread.
     2. Jobs are added asyncronously with job_queue_add_job_mt() from othread threads(s).


   Unfortunately it does not work properly (i.e. Ctrl-C breaks) to use a Python
   thread to invoke the job_queue_run_jobs() function; and this function is
   mainly a workaround around that problem. The function will create a new
   thread and run job_queue_run_jobs() in that thread; the calling thread will
   just return.

   No reference is retained to the thread actually running the
   job_queue_run_jobs() function.
*/


void job_queue_run_jobs_threaded(job_queue_type * queue , int num_total_run, bool verbose) {
  pthread_t        queue_thread;
  job_queue_start_manager_thread( queue , &queue_thread , num_total_run , verbose );
  pthread_detach( queue_thread );             /* Signal that the thread resources should be cleaned up when
                                                 the thread has exited. */
}



/*****************************************************************/
/* Adding new jobs - it is complicated ... */


/**
   This initializes the non-driver-spesific fields of a job, i.e. the
   name, runpath and so on, and sets the job->status ==
   JOB_QUEUE_WAITING. This status means the job is ready to be
   submitted proper to one of the drivers (when a slot is ready).
   When submitted the job will get (driver specific) job_data != NULL
   and status SUBMITTED.
*/




int job_queue_add_job(job_queue_type * queue ,
                      const char * run_cmd ,
                      job_callback_ftype * done_callback,
                      job_callback_ftype * retry_callback,
                      job_callback_ftype * exit_callback,
                      void * callback_arg ,
                      int num_cpu ,
                      const char * run_path ,
                      const char * job_name ,
                      int argc ,
                      const char ** argv) {

  //Fail hard if queue is not open
  job_queue_check_open(queue);

  if (!queue->user_exit) {/* We do not accept new jobs if a user-shutdown has been iniated. */
    int queue_index;
    {
      job_queue_node_type * node = job_queue_node_alloc( );
      job_queue_node_initialize( node , run_path , num_cpu , job_name , argc , argv );
      job_queue_node_set_exit_file( node , queue->exit_file );
      job_queue_node_set_ok_file( node , queue->ok_file );
      job_queue_node_set_cmd( node , run_cmd );
      job_queue_node_init_callbacks( node , exit_callback , retry_callback , done_callback , callback_arg );

      job_list_get_wrlock( queue->job_list );
      {
        job_list_add_job( queue->job_list , node );
        queue_index = job_queue_node_get_queue_index(node);
        queue->status_list[ STATUS_INDEX( job_queue_node_get_status(node)) ]++;
      }
      job_list_unlock( queue->job_list );

      job_queue_change_node_status(queue , node , JOB_QUEUE_WAITING);
    }
    return queue_index;   /* Handle used by the calling scope. */
  } else
    return -1;
}




/**
   When the job_queue_run_jobs() has been called with @total_num_jobs
   == 0 that means that the total number of jobs to run is not known
   in advance. In that case it is essential to signal the queue when
   we will not submit any more jobs, so that it can finalize and
   return. That is done with the function job_queue_submit_complete()
*/

void job_queue_submit_complete( job_queue_type * queue ){
  queue->submit_complete = true;
}



/**
   The calling scope must retain a handle to the current driver and
   free it.  Should (in principle) be possible to change driver on a
   running system whoaaa. Will read and update the max_running value
   from the driver.
*/

void job_queue_set_driver(job_queue_type * queue , queue_driver_type * driver) {
  queue->driver = driver;
}


bool job_queue_has_driver(const job_queue_type * queue ) {
  if (queue->driver == NULL)
    return false;
  else
    return true;
}


job_driver_type job_queue_lookup_driver_name( const char * driver_name ) {
  if (strcmp( driver_name , "LOCAL") == 0)
    return LOCAL_DRIVER;
  else if (strcmp( driver_name , "RSH") == 0)
    return RSH_DRIVER;
  else if (strcmp( driver_name , "LSF") == 0)
    return LSF_DRIVER;
  else {
    util_abort("%s: driver:%s not recognized \n",__func__ , driver_name);
    return NULL_DRIVER;
  }
}

/*****************************************************************/



void job_queue_set_max_submit( job_queue_type * job_queue , int max_submit ) {
  job_queue->max_submit = max_submit;
}


int job_queue_get_max_submit(const job_queue_type * job_queue ) {
  return job_queue->max_submit;
}




/**
   Observe that the job_queue returned by this function is NOT ready
   for use; a driver must be set explicitly with a call to
   job_queue_set_driver() first.
*/

job_queue_type * job_queue_alloc(int  max_submit               ,
                                 const char * ok_file ,
                                 const char * exit_file ) {



  job_queue_type * queue  = util_malloc(sizeof * queue );
  queue->usleep_time      = 250000; /* 1000000 : 1 second */
  queue->max_ok_wait_time = 60;
  queue->max_duration     = 0;
  queue->stop_time        = 0;
  queue->max_submit       = max_submit;
  queue->driver           = NULL;
  queue->ok_file          = util_alloc_string_copy( ok_file );
  queue->exit_file        = util_alloc_string_copy( exit_file );
  queue->open             = true;
  queue->user_exit        = false;
  queue->pause_on         = false;
  queue->running          = false;
  queue->grow             = false;
  queue->submit_complete  = false;
  queue->work_pool        = NULL;
  queue->job_list         = job_list_alloc(  );

  pthread_mutex_init( &queue->status_mutex , NULL);
  pthread_mutex_init( &queue->submit_mutex  , NULL);
  pthread_mutex_init( &queue->run_mutex    , NULL );
  pthread_rwlock_init( &queue->queue_lock , NULL);

  job_queue_clear_status( queue );
  return queue;
}

/**
   Returns true if the queue is currently paused, which means that no
   more jobs are submitted.
*/


bool job_queue_get_pause( const job_queue_type * job_queue ) {
  return job_queue->pause_on;
}


void job_queue_set_pause_on( job_queue_type * job_queue) {
  job_queue->pause_on = true;
}


void job_queue_set_pause_off( job_queue_type * job_queue) {
  job_queue->pause_on = false;
}


void * job_queue_iget_job_data( job_queue_type * job_queue , int job_nr ) {
  job_queue_node_type * job = job_queue_iget_node( job_queue, job_nr );
  return job_queue_node_get_data( job );
}



void job_queue_free(job_queue_type * queue) {
  util_safe_free( queue->ok_file );
  util_safe_free( queue->exit_file );
  job_list_free( queue->job_list );
  free(queue);
}


/*****************************************************************/

const char * job_queue_status_name( job_status_type status ) {
  return status_name[ STATUS_INDEX(status) ];
}


/*****************************************************************/

job_status_type job_queue_get_status( queue_driver_type * driver , job_queue_node_type * job) {
  return queue_driver_get_status( driver , job_queue_node_get_data(job));
}

