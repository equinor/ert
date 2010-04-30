#define  _GNU_SOURCE   /* Must define this to get access to pthread_rwlock_t */
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <job_queue.h>
#include <msg.h>
#include <util.h>
#include <basic_queue_driver.h>
#include <pthread.h>
#include <unistd.h>
#include <arg_pack.h>

/**
   The running of external jobs is handled thrugh an abstract
   job_queue implemented in this file; the job_queue then contains a
   'driver' which actually runs the job. All drivers must support the
   following functions

     submit: This will submit a job, and return a pointer to a 
             newly allocated queue_job instance.

     clean:  This will clear up all resources used by the job.

     abort:  This will stop the job, and then call clean.

     status: This will get the status of the job. 

     display_info: [Optional] - this is only for the LSF layer to
                   display the name of a possibly faulty node before 
                   restarting.

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
        basic_queue_driver_type instance which is a struct like this:

        struct basic_queue_driver_struct {
            UTIL_TYPE_ID_DECLARATION
            QUEUE_DRIVER_FUNCTIONS
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
      like e.g. ECLIPSE and RMS. The job_script does not realiably
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

         
   Observe that there is a random ugly coupling between this file and
   the job script thorugh the name of the EXIT and OK files. In this
   file those names are just defined with #define below here. If these
   are changed, all hell will break loose.
*/


#define EXIT_FILE "EXIT"
#define OK_FILE   "OK"


typedef enum {SUBMIT_OK           = 0 , 
              SUBMIT_JOB_FAIL     = 1 , /* Typically no more attempts. */
              SUBMIT_DRIVER_FAIL  = 2 , /* The driver would not take the job - for whatever reason?? */ 
              SUBMIT_QUEUE_PAUSED = 3 } /* The queue is currently not accepting more jobs. */   submit_status_type;



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

typedef struct {
  bool                   user_kill;       /* Used by the external scope to say that this job should be killed - the actual killing should handled by the same thread as running the jobs. */
  int                  	 submit_attempt;  /* Which attempt is this ... */
  char                  *exit_file;       /* The queue will look for the occurence of this file to detect a failure. */
  char                  *ok_file;         /* The queue will look for this file to verify that the job was OK - can be NULL - in which case it is ignored. */
  char                 	*job_name;        /* The name of the job. */
  char                  *run_path;        /* Where the job is run - absolute path. */
  job_status_type  	 job_status;      /* The current status of the job. */
  void           	*job_data;        /* Driver specific data about this job - fully handled by the driver. */
  const void            *job_arg;         /* Untyped data which is sent to the submit function as extra argument - can be whatever - fully owned by external scope.*/
  time_t                 submit_time;     /* When was the job added to job_queue - the FIRST TIME. */
  time_t                 sim_start;       /* When did the job change status -> RUNNING - the LAST TIME. */
  pthread_rwlock_t       job_lock;
} job_queue_node_type;



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
  int                        size;          			/* The total number of job slots in the queue. */
  int                        max_submit;    			/* The maximum number of submit attempts for one job. */
  int                        max_running;   			/* The maximum number of concurrently running jobs. */
  char                     * run_cmd;       			/* The command which is run (i.e. path to an executable with arguments). */
  job_queue_node_type     ** jobs;          			/* A vector of job nodes .*/
  basic_queue_driver_type  * driver;        			/* A pointer to a driver instance (LSF|LOCAL|RSH) which actually 'does it'. */
  int                        status_list[JOB_QUEUE_MAX_STATE];  /* The number of jobs in the different states. */
  bool                       running;                           /* Is the job queue currently running? */  
  bool                       pause_on;                          /* Is the job currently paused? If pause_on == true no new jobs will be submitted. */ 
  bool                       user_exit;                         /* If there comes an external signal to abondond the whole thing user_exit will be set to true, and things start to dwindle down. */ 
  unsigned long              usleep_time;                       /* The sleep time before checking for updates. */
  pthread_rwlock_t           active_rwlock;
  pthread_rwlock_t           status_rwlock;                  
};

/*****************************************************************/

static int STATUS_INDEX( job_status_type status ) {
  int status_index = 0;
  while ( (status != 1) && (status_index < JOB_QUEUE_MAX_STATE)) {
    status >>= 1;
    status_index++;
  }
  if (status != 1)
    util_abort("%s: failed to get index from:%d \n",__func__ , status);
  return status_index;
}



/*****************************************************************/

static void job_queue_node_clear(job_queue_node_type * node) {
  node->job_status     = JOB_QUEUE_NOT_ACTIVE;
  node->submit_attempt = 0;
  node->job_name       = NULL;
  node->run_path       = NULL;
  node->job_data       = NULL;
  node->exit_file      = NULL;
  node->ok_file        = NULL;
  node->user_kill      = false;  
}


static job_queue_node_type * job_queue_node_alloc() {
  job_queue_node_type * node = util_malloc(sizeof * node , __func__);
  job_queue_node_clear(node);
  pthread_rwlock_init( &node->job_lock , NULL);
  return node;
}


static void job_queue_node_set_status(job_queue_node_type * node, job_status_type status) {
  node->job_status = status;
}

static void job_queue_node_free_data(job_queue_node_type * node) {
  util_safe_free(node->run_path);  node->run_path  = NULL;
  util_safe_free(node->job_name);  node->job_name  = NULL;
  util_safe_free(node->exit_file); node->exit_file = NULL;
  util_safe_free(node->ok_file);   node->ok_file   = NULL;
  if (node->job_data != NULL) 
    util_abort("%s: internal error - driver spesific job data has not been freed - will leak.\n",__func__);
}


static void job_queue_node_free(job_queue_node_type * node) {
  job_queue_node_free_data(node);
  free(node);
}

static job_status_type job_queue_node_get_status(const job_queue_node_type * node) {
  return node->job_status;
}




static void job_queue_node_finalize(job_queue_node_type * node) {
  job_queue_node_free_data(node);
  job_queue_node_clear(node);
}



/*****************************************************************/

static bool job_queue_change_node_status(job_queue_type *  , job_queue_node_type *  , job_status_type );


static void job_queue_initialize_node(job_queue_type * queue , const char * run_path , const char * job_name , int job_index , const void * job_arg) {
  job_queue_node_type * node = queue->jobs[job_index];
  node->submit_attempt = 0;
  node->job_name       = util_alloc_string_copy( job_name );
  node->job_data       = NULL;
  node->job_arg        = job_arg;
  
  if (util_is_abs_path(run_path)) 
    node->run_path = util_alloc_string_copy( run_path );
  else
    node->run_path = util_alloc_realpath( run_path );
  
  if ( !util_is_directory(node->run_path) ) 
    util_abort("%s: the run_path: %s does not exist - aborting \n",__func__ , node->run_path);
  
  node->exit_file   = util_alloc_filename(node->run_path , EXIT_FILE , NULL);
  node->ok_file     = util_alloc_filename(node->run_path , OK_FILE   , NULL);
  node->sim_start   = -1;
  node->submit_time = time( NULL );
  job_queue_change_node_status(queue , node , JOB_QUEUE_WAITING);   /* Now the job is ready to be picked by the queue manager. */
}




static void job_queue_assert_queue_index(const job_queue_type * queue , int queue_index) {
  if (queue_index < 0 || queue_index >= queue->size) 
    util_abort("%s: invalid queue_index - internal error - aborting \n",__func__);
}



static bool job_queue_change_node_status(job_queue_type * queue , job_queue_node_type * node , job_status_type new_status) {
  bool status_change = false;
  pthread_rwlock_wrlock( &queue->status_rwlock );
  {
    job_status_type old_status = job_queue_node_get_status(node);
    job_queue_node_set_status(node , new_status);
    queue->status_list[ STATUS_INDEX(old_status) ]--;
    queue->status_list[ STATUS_INDEX(new_status) ]++;

    if (new_status != old_status) {
      status_change = true;
      if (new_status == JOB_QUEUE_RUNNING) 
        node->sim_start = time( NULL );
    }
    
  }
  pthread_rwlock_unlock( &queue->status_rwlock );
  return status_change;
}


static void job_queue_free_job(job_queue_type * queue , job_queue_node_type * node) {
  basic_queue_driver_type *driver  = queue->driver;
  driver->free_job(driver , node->job_data);
  node->job_data = NULL;
}



/**
   Observe that this function should only query the driver for state
   change when the job is currently in one of the states: 

     JOB_QUEUE_WAITING || JOB_QUEUE_PENDING || JOB_QUEUE_RUNNING 

   The other state transitions are handled by the job_queue itself,
   without consulting the driver functions.
*/

static void job_queue_kill_job__( job_queue_type * queue , int queue_index);

static void job_queue_update_status(job_queue_type * queue ) {
  basic_queue_driver_type *driver  = queue->driver;
  int ijob;
  for (ijob = 0; ijob < queue->size; ijob++) {
    job_queue_node_type * node = queue->jobs[ijob];
    if (node->job_data != NULL) {
      if (node->user_kill)
        job_queue_kill_job__( queue , ijob);     /* External scope has marked this job for a kill - now we react on it. */
      else {
        job_status_type current_status = job_queue_node_get_status(node);
        if ((current_status == JOB_QUEUE_RUNNING) || (current_status == JOB_QUEUE_WAITING) || (current_status == JOB_QUEUE_PENDING)) {
          job_status_type new_status = driver->get_status(driver , node->job_data);
          job_queue_change_node_status(queue , node , new_status);
        }
      }
    }
  }
}



static submit_status_type job_queue_submit_job(job_queue_type * queue , int queue_index) {
  submit_status_type submit_status;
  if (queue->pause_on)
    submit_status = SUBMIT_QUEUE_PAUSED;   /* The queue is currently not accepting more jobs. */
  else {
    job_queue_assert_queue_index(queue , queue_index);
    {
      job_queue_node_type     * node    = queue->jobs[queue_index];
      basic_queue_driver_type * driver  = queue->driver;
      
      if (node->submit_attempt < queue->max_submit) {
        void * job_data = driver->submit( queue->driver  , 
                                          queue_index    , 
                                          queue->run_cmd , 
                                          node->run_path , 
                                          node->job_name , 
                                          node->job_arg );
        
        if (job_data != NULL) {
          node->job_data = job_data;
          node->submit_attempt++;
          job_queue_change_node_status(queue , node , JOB_QUEUE_WAITING ); /* This is when it is installed as runnable in the internal queue   */ 
                                                                           /* The update_status function will grab this and update to running. */
          submit_status = SUBMIT_OK;
        } else
          submit_status = SUBMIT_DRIVER_FAIL;
      } else {
        /* 
           The queue system will not make more attempts to submit this job. The
           external scope might call job_queue_set_external_restart() - otherwise
           this job is failed.
        */
        job_queue_change_node_status(queue , node , JOB_QUEUE_RUN_FAIL);
        submit_status = SUBMIT_JOB_FAIL;
      }
    }
  }
  return submit_status;
}










job_status_type job_queue_get_job_status(job_queue_type * queue , int job_index) {
  job_queue_node_type * node = queue->jobs[job_index];
  return node->job_status;
}


/**
   This function will copy the internal queue->status_list to the
   input variable @external_status_list. The external scope can then
   query this variable with the enum fields defined in
   'basic_queue_driver.h':

      #include <basic_queue_driver.h>

      int * status_list = util_malloc( sizeof * status_list * JOB_QUEUE_MAX_STATE , __func__);
   
      job_queue_export_status_summary( queue , status_list );
      printf("Running jobs...: %03d \n", status_list[ JOB_QUEUE_RUNNING ]);
      printf("Waiting jobs:..: %03d \n", status_list[ JOB_QUEUE_WAITING ]);

   Alternatively the function job_queue_iget_status_summary() can be used
*/



void job_queue_export_status_summary( job_queue_type * queue , int * external_status_list) {
  pthread_rwlock_rdlock( &queue->status_rwlock );
  {
    memcpy( external_status_list , queue->status_list , JOB_QUEUE_MAX_STATE * sizeof * queue->status_list );
  }
  pthread_rwlock_unlock( &queue->status_rwlock );
}



/**
   Will return the number of jobs with status @status.

      #include <basic_queue_driver.h>

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




void job_queue_set_load_OK(job_queue_type * queue , int job_index) {
  job_queue_node_type * node = queue->jobs[job_index];
  job_queue_change_node_status( queue , node , JOB_QUEUE_ALL_OK);
}



void job_queue_set_all_fail(job_queue_type * queue , int job_index) {
  job_queue_node_type * node = queue->jobs[job_index];
  job_queue_change_node_status( queue , node , JOB_QUEUE_ALL_FAIL);
}

/**
   Observe that jobs with status JOB_QUEUE_WAITING can also be killed; for those
   jobs the kill should be interpreted as "Forget about this job for now and set
   the status JOB_QUEUE_USER_KILLED", however it is important that we not call
   the driver->kill() function on it because the job slot will have no data
   (i.e. LSF jobnr), and the driver->kill() function will fail if presented with
   such a job.

   Observe that jobs (slots) with status JOB_QUEUE_NOT_ACTIVE can NOT be
   meaningfully killed; that is because these jobs have not yet been submitted
   to the queue system, and there is not yet established a mapping between
   external id and queue_index.

   Observe that this function changes the jobs table of the queue, and should
   ONLY be invoked by the thread running the jobs with job_queue_run_jobs()
   function.
*/

static void job_queue_kill_job__( job_queue_type * queue , int job_index) {
  job_queue_node_type * node = queue->jobs[job_index];
  if (node->job_status & JOB_QUEUE_CAN_KILL) {
    basic_queue_driver_type * driver = queue->driver;
    
    job_queue_change_node_status( queue , node , JOB_QUEUE_USER_KILLED );
    if (node->job_status != JOB_QUEUE_NOT_ACTIVE)
      driver->kill_job( driver , node->job_data );
    node->job_data = NULL;               
  }
  node->user_kill = false;
}


/**
   Observe that the driver->kill() function is responsible for freeing
   the driver specific data. Only jobs which have a status matching
   "JOB_QUEUE_CAN_KILL" can be killed; if the job is not in a killable
   state the function will do nothing. This includes trying to kill a
   job which is not even found.
   
   The function will return true if the job is actually killed, and
   false otherwise.
*/

bool job_queue_kill_job( job_queue_type * queue , int job_index) {
  job_queue_node_type * node = queue->jobs[ job_index ];
  if (node->job_status & JOB_QUEUE_CAN_KILL) {
    node->user_kill = true;                             /* We signal that this job should be killed - the actual action to do it 
                                                           is initiated from the job_queue_update_status() function. */
    return true;
  } else
    return false;
}


/**
   The external scope asks the queue to restart the the job. We reset
   the submit counter to zero.
*/
   
void job_queue_set_external_restart(job_queue_type * queue , int job_index) {
  job_queue_node_type * node = queue->jobs[job_index];
  node->submit_attempt       = 0;
  job_queue_change_node_status( queue , node , JOB_QUEUE_WAITING );
}


/**
   The external scope has decided that no more attempts will be tried. This job
   should really fail!! 
*/
void job_queue_set_external_fail(job_queue_type * queue , int job_index) {
  job_queue_node_type * node = queue->jobs[job_index];
  node->submit_attempt       = 0;
  job_queue_change_node_status( queue , node , JOB_QUEUE_ALL_FAIL);
}

/**
   The external scope is loading results. Just for keeping track of status.
*/

void job_queue_set_external_load(job_queue_type * queue , int job_index) {
  job_queue_node_type * node = queue->jobs[job_index];
  job_queue_change_node_status( queue , node , JOB_QUEUE_LOADING );
}



time_t job_queue_iget_sim_start( job_queue_type * queue, int job_index) {
  job_queue_node_type * node = queue->jobs[job_index];
  return node->sim_start;
}



time_t job_queue_iget_submit_time( job_queue_type * queue, int job_index) {
  job_queue_node_type * node = queue->jobs[job_index];
  return node->submit_time;
}




static void job_queue_print_jobs(const job_queue_type *queue) {
  int waiting  = queue->status_list[ STATUS_INDEX(JOB_QUEUE_WAITING) ];
  int pending  = queue->status_list[ STATUS_INDEX(JOB_QUEUE_PENDING) ];
  
    /* 
       EXIT and DONE are included in "xxx_running", because the target
       file has not yet been checked.
    */
  int running  = queue->status_list[ STATUS_INDEX(JOB_QUEUE_RUNNING) ] + queue->status_list[ STATUS_INDEX(JOB_QUEUE_DONE) ] + queue->status_list[ STATUS_INDEX(JOB_QUEUE_EXIT) ];
  int complete = queue->status_list[ STATUS_INDEX(JOB_QUEUE_ALL_OK) ];
  int failed   = queue->status_list[ STATUS_INDEX(JOB_QUEUE_ALL_FAIL) ];
  int loading  = queue->status_list[ STATUS_INDEX(JOB_QUEUE_RUN_OK) ];  
  
  printf("Waiting: %3d    Pending: %3d    Running: %3d     Loading: %3d    Failed: %3d   Complete: %3d   [ ]\b",waiting , pending , running , loading , failed , complete);
  fflush(stdout);
}


static void job_queue_display_job_info( const job_queue_type * job_queue , const job_queue_node_type * job_node ) {
  if (job_queue->driver->display_info != NULL)
    job_queue->driver->display_info( job_queue->driver , job_node->job_data );
  printf("\n");
}


/**
   
*/
void job_queue_init( job_queue_type * job_queue ) {
}

/** 
    This function goes through all the nodes and call finalize on
    them. It is essential that this routine is not called before all
    the jobs have completed. It is also essential that it is called
    before the job_queue is reused for a new set of simulations.
*/

void job_queue_finalize(job_queue_type * queue) {
  int i;
  
  pthread_rwlock_wrlock( &queue->active_rwlock );
  
  for (i=0; i < queue->size; i++) 
    job_queue_node_finalize(queue->jobs[i]);
  
  for (i=0; i < JOB_QUEUE_MAX_STATE; i++) 
    queue->status_list[i] = 0;

  queue->user_exit = false;
  pthread_rwlock_unlock( &queue->active_rwlock );
}


bool job_queue_is_running( const job_queue_type * queue ) {
  return queue->running;
}



void job_queue_run_jobs(job_queue_type * queue , int num_total_run, bool verbose) {
  queue->running = true;
  if (num_total_run > queue->size) util_abort("%s: invalid num_total_run \n",__func__);
  {

    const int max_ok_wait_time = 60; /* Seconds to wait for an OK file - when the job itself has said all OK. */
    const int ok_sleep_time    =  1; /* Time to wait between checks for OK|EXIT file.                         */

    msg_type * submit_msg = NULL;
    bool new_jobs         = false;
    bool cont             = true;
    int  phase = 0;
    int  old_status_list[ JOB_QUEUE_MAX_STATE ];
    {
      int i;
      for (i=0; i < JOB_QUEUE_MAX_STATE; i++)
        old_status_list[i] = -1;
    }

    if (verbose)
      submit_msg = msg_alloc("Submitting new jobs:  ]");
  
    do {
      char spinner[4];
      spinner[0] = '-';
      spinner[1] = '\\';
      spinner[2] = '|';
      spinner[3] = '/';

      if (queue->user_exit) { /* An external thread has called the job_queue_user_exit() function, and we
                                 should kill all jobs, do some clearing up and go home. */
        int queue_index;
        for (queue_index = 0; queue_index < queue->size; queue_index++) {
          job_queue_kill_job__( queue , queue_index);
          job_queue_change_node_status( queue , queue->jobs[queue_index] , JOB_QUEUE_USER_EXIT);
        }
        cont = false;
      } 
        
      if (cont) {
        job_queue_update_status( queue );
        if ( (memcmp(old_status_list , queue->status_list , JOB_QUEUE_MAX_STATE * sizeof * old_status_list) != 0) || new_jobs ) {
          if (verbose) {
            printf("\b \n");
            job_queue_print_jobs(queue);
          }
          memcpy(old_status_list , queue->status_list , JOB_QUEUE_MAX_STATE * sizeof * old_status_list);
        } 
        
        if ((queue->status_list[ STATUS_INDEX(JOB_QUEUE_ALL_OK)    ] + 
             queue->status_list[ STATUS_INDEX(JOB_QUEUE_ALL_FAIL)  ] +
             queue->status_list[ STATUS_INDEX(JOB_QUEUE_USER_EXIT) ]) == num_total_run)
          cont = false;
        
        if (cont) {
          if (verbose) {
            printf("\b%c",spinner[phase]); 
            fflush(stdout);
            phase = (phase + 1) % 4;
          }
          
          {
            
            /* Submitting new jobs */
            int max_submit     = 5; /* This is the maximum number of jobs submitted in one while() { ... } below. 
                                       Only to ensure that the waiting time before a status update is not to long. */
            int total_active   = queue->status_list[ STATUS_INDEX(JOB_QUEUE_PENDING) ] + queue->status_list[ STATUS_INDEX(JOB_QUEUE_RUNNING) ];
            int num_submit_new = util_int_min( max_submit , queue->max_running - total_active );
            char spinner2[2];
            spinner2[1] = '\0';
            
            new_jobs = false;
            if (queue->status_list[ STATUS_INDEX(JOB_QUEUE_WAITING) ] > 0)   /* We have waiting jobs at all           */
              if (num_submit_new > 0)                                        /* The queue can allow more running jobs */
                new_jobs = true;
            
            
            if (new_jobs) {
              int submit_count = 0;
              int queue_index  = 0;
              
              while ((queue_index < queue->size) && (num_submit_new > 0)) {
                job_queue_node_type * node = queue->jobs[queue_index];
                if (job_queue_node_get_status(node) == JOB_QUEUE_WAITING) {
                  {
                    submit_status_type submit_status = job_queue_submit_job(queue , queue_index);
                    
                    if (submit_status == SUBMIT_OK) {
                      if ((submit_count == 0) && verbose) {
                        printf("\b");
                        msg_show(submit_msg);
                        printf("\b\b");
                      }
                      spinner2[0] = spinner[phase];
                      msg_update(submit_msg , spinner2);
                      phase = (phase + 1) % 4;
                      num_submit_new--;
                      submit_count++;
                    } else if ((submit_status == SUBMIT_DRIVER_FAIL) || (submit_status == SUBMIT_QUEUE_PAUSED))
                      break;
                  }
                }
                queue_index++;
              }
              
              if ((submit_count > 0) && verbose) {
                printf("  "); fflush(stdout);
                msg_hide(submit_msg);
                printf(" ]\b"); fflush(stdout);
              } else 
                /* 
                   We wanted to - and tried - to submit new jobs; but the
                   driver failed to deliver.
                */
                new_jobs = false;
            }
          }
          
          {
            /*
              Checking for complete / exited jobs.
            */
            int queue_index;
            for (queue_index = 0; queue_index < queue->size; queue_index++) {
              job_queue_node_type * node = queue->jobs[queue_index];
              switch ( job_queue_node_get_status(node) ) {
              case(JOB_QUEUE_DONE):
                if (util_file_exists(node->exit_file)) 
                  job_queue_change_node_status(queue , node , JOB_QUEUE_EXIT);
                else {
                  /* Check if the OK file has been produced. Wait and retry. */
                  if (node->ok_file != NULL) {
                    int  total_wait_time = 0;
                    
                    while (true) {
                      if (util_file_exists( node->ok_file )) {
                        job_queue_change_node_status(queue , node , JOB_QUEUE_RUN_OK);
                        job_queue_free_job(queue , node);  /* This frees the storage allocated by the driver - the storage allocated by the queue layer is retained. */
                        break;
                      } else {
                        if (total_wait_time <  max_ok_wait_time) {
                          sleep( ok_sleep_time );
                          total_wait_time += ok_sleep_time;
                        } else {
                          /* We have waited long enough - this does not seem to give any OK file. */
                          job_queue_change_node_status(queue , node , JOB_QUEUE_EXIT);
                          break;
                        }
                      }
                    } 
                  } else {
                    /* We have not set the ok_file - then we just assume that the job is OK. */
                    job_queue_change_node_status(queue , node , JOB_QUEUE_RUN_OK);
                    job_queue_free_job(queue , node);   /* This frees the storage allocated by the driver - the storage allocated by the queue layer is retained. */
                  }
                }
                break;
              case(JOB_QUEUE_EXIT):
                if (verbose) {
                  printf("Job: %s failed | ",node->job_name);
                  job_queue_display_job_info( queue , node );
                }
                /* 
                   If the job has failed with status JOB_QUEUE_EXIT it
                   will always go via status JOB_QUEUE_WAITING first. The
                   job dispatched will then either resubmit, in case there
                   are more attempts to go, or set the status to
                   JOB_QUEUE_FAIL.
                */
                job_queue_change_node_status(queue , node , JOB_QUEUE_WAITING);
                job_queue_free_job(queue , node);  /* This frees the storage allocated by the driver - the storage allocated by the queue layer is retained. */
                break;
              default:
                break;
              }
            }
          }
          
          if (!new_jobs)
            usleep(queue->usleep_time);
          /*
            else we have submitted new jobs - and know the status is out
            of sync, no need to wait before a rescan.
          */
        }
      }
    } while ( cont );

    if (verbose) {
      printf("\n");
      msg_free(submit_msg , false);
    }
  }
  queue->user_exit = false;
  queue->running   = false;
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
  return NULL;
}



void job_queue_insert_job(job_queue_type * queue , const char * run_path , const char * job_name , int job_index , const void * job_arg) {
  pthread_rwlock_wrlock( &queue->active_rwlock );
  {
    job_queue_initialize_node(queue , run_path , job_name , job_index , job_arg);
  }
  pthread_rwlock_unlock( &queue->active_rwlock );
}




/**
   Should (in principle) be possible to change driver on a running system whoaaa.
*/

void job_queue_set_driver(job_queue_type * queue , basic_queue_driver_type * driver) {
  if (queue->driver != NULL) {
    basic_queue_driver_type * old_driver = queue->driver;
    old_driver->free_driver(old_driver);
  }
  
  queue->driver = driver;
}


/**
   Observe that if the max number of running jobs is decreased,
   nothing will be done to reduce the number of jobs currently
   running; but no more jobs will be submitted until the number of
   running has fallen below the new limit.
*/

void job_queue_set_max_running( job_queue_type * queue , int max_running ) {
  queue->max_running = max_running;
  if (queue->max_running < 0)
    queue->max_running = 0;
}

/*
  The return value is the new value for max_running.
*/
int job_queue_inc_max_runnning( job_queue_type * queue, int delta ) {
  job_queue_set_max_running( queue , queue->max_running + delta );
  return queue->max_running;
}

int job_queue_get_max_running( const job_queue_type * queue ) {
  return queue->max_running;
}


job_driver_type job_queue_get_driver_type( const job_queue_type * queue ) {
  if (queue->driver == NULL)
    return NULL_DRIVER;
  else
    return queue->driver->driver_type;
}

/*****************************************************************/

const char * job_queue_get_driver_name( const job_queue_type * queue ) {
  job_driver_type driver_type = job_queue_get_driver_type( queue );
  switch( driver_type ) {
  case(LSF_DRIVER):
    return "LSF";
    break;
  case(RSH_DRIVER):
    return "RSH";
    break;
  case(LOCAL_DRIVER):
    return "LOCAL";
    break;
  default:
    util_abort("%s: driver_type not set?? \n",__func__);
    return NULL;
  }
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


void job_queue_set_size( job_queue_type * queue , int size ) {
  /* Delete the existing nodes. */
  {
    if (queue->size != 0) {
      for (int i=0; i < queue->size; i++)
        job_queue_node_free( queue->jobs[i] );
    }
  }
  
  
  /* Fill the new nodes */
  {
    queue->size            = size;
    queue->jobs            = util_realloc(queue->jobs , size * sizeof * queue->jobs , __func__);
    {
      int i;
      for (i=0; i < size; i++) 
        queue->jobs[i] = job_queue_node_alloc();

      
      /** 
          Here the status list is clearead, and then it is set/updated
          according to the status of the nodes. This is the only place
          there is a net change in the status list.
      */
      
      for (i=0; i < JOB_QUEUE_MAX_STATE; i++)
        queue->status_list[i] = 0;
      
      for (i=0; i < size; i++) 
        queue->status_list[job_queue_node_get_status(queue->jobs[i])]++;
      
    }
  }
}


void job_queue_set_run_cmd( job_queue_type * job_queue , const char * run_cmd ) {
  job_queue->run_cmd = util_realloc_string_copy( job_queue->run_cmd , run_cmd );
}

const char * job_queue_get_run_cmd( job_queue_type * job_queue) {
  return job_queue->run_cmd;
}


void job_queue_set_max_submit( job_queue_type * job_queue , int max_submit ) {
  job_queue->max_submit = max_submit;
}


int job_queue_get_max_submit(const job_queue_type * job_queue ) {
  return job_queue->max_submit;
}



/**
   The job_queue instance can be resized afterwards with a call to
   job_queue_set_size(). Observe that the job_queue returned by this
   function is NOT ready for use; a driver must be set explicitly with
   a call to job_queue_set_driver() first.
*/

job_queue_type * job_queue_alloc(int size , 
                                 int max_running , 
                                 int max_submit , 
				 const char * run_cmd) {
				 

  job_queue_type * queue = util_malloc(sizeof * queue , __func__);
  queue->running         = false;
  queue->jobs            = NULL;
  queue->usleep_time     = 1000000; /* 1 second */
  queue->max_running     = max_running;
  queue->max_submit      = max_submit;
  queue->driver          = NULL;
  queue->run_cmd         = NULL;
  queue->pause_on        = false;
  queue->user_exit       = false;
  job_queue_set_run_cmd( queue , run_cmd );
  job_queue_set_size( queue , size );
  pthread_rwlock_init( &queue->active_rwlock , NULL);
  pthread_rwlock_init( &queue->status_rwlock , NULL);
  return queue;
}

/**
   Returns true if the queue is currently paused, which means that no
   more jobs are submitted. 
*/


bool job_queue_get_pause( const job_queue_type * job_queue ) {
  return job_queue->pause_on;
}

static void job_queue_set_pause__( job_queue_type * job_queue, bool pause_on) {
  job_queue->pause_on = pause_on;
}

void job_queue_set_pause_on( job_queue_type * job_queue) {
  job_queue_set_pause__( job_queue , true );
}


void job_queue_set_pause_off( job_queue_type * job_queue) {
  job_queue_set_pause__( job_queue , false);
}





void job_queue_free(job_queue_type * queue) {
  free(queue->run_cmd);
  {
    int i;
    for (i=0; i < queue->size; i++) 
      job_queue_node_free(queue->jobs[i]);
    free(queue->jobs);
  }
  {
    basic_queue_driver_type * driver = queue->driver;
    driver->free_driver(driver);
  }
  free(queue);
  queue = NULL;
}


/*****************************************************************/

const char * job_queue_status_name( job_status_type status ) {
  switch (status) {
  case(JOB_QUEUE_NOT_ACTIVE):
    return "JOB_QUEUE_NOT_ACTIVE";
    break;
  case(JOB_QUEUE_LOADING):
    return "JOB_QUEUE_LOADING";
    break;
  case(JOB_QUEUE_WAITING):
    return "JOB_QUEUE_WAITING";
    break;
  case(JOB_QUEUE_PENDING):
    return "JOB_QUEUE_PENDING";
    break;
  case(JOB_QUEUE_RUNNING):
    return "JOB_QUEUE_RUNNING";
    break;
  case(JOB_QUEUE_DONE):
    return "JOB_QUEUE_DONE";
    break;
  case(JOB_QUEUE_EXIT):
    return "JOB_QUEUE_EXIT";
    break;
  case(JOB_QUEUE_RUN_OK):
    return "JOB_QUEUE_RUN_OK";
    break;
  case(JOB_QUEUE_RUN_FAIL):
    return "JOB_QUEUE_RUN_FAIL";
    break;
  case(JOB_QUEUE_ALL_OK):
    return "JOB_QUEUE_ALL_OK";
    break;
  case(JOB_QUEUE_ALL_FAIL):
    return "JOB_QUEUE_ALL_FAIL";
    break;
  case(JOB_QUEUE_USER_KILLED):
    return "JOB_QUEUE_USER_KILLED";
    break;
  default:
    util_abort("%s: invalid job_status value:%d \n",__func__ , status);
    return NULL;
  }
}
