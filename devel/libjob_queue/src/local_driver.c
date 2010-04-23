#include <stdlib.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <basic_queue_driver.h>
#include <local_driver.h>
#include <util.h>
#include <pthread.h>
#include <arg_pack.h>
#include <errno.h>

struct local_job_struct {
  UTIL_TYPE_ID_DECLARATION
  bool       	  active;
  job_status_type status;
  pthread_t       run_thread;
};


#define LOCAL_DRIVER_TYPE_ID 66196305
#define LOCAL_JOB_TYPE_ID    63056619

struct local_driver_struct {
  UTIL_TYPE_ID_DECLARATION
  QUEUE_DRIVER_FUNCTIONS
  pthread_attr_t     thread_attr;
  pthread_mutex_t    submit_lock;
};

/*****************************************************************/


static UTIL_SAFE_CAST_FUNCTION( local_driver , LOCAL_DRIVER_TYPE_ID )
static UTIL_SAFE_CAST_FUNCTION( local_job , LOCAL_JOB_TYPE_ID )




local_job_type * local_job_alloc() {
  local_job_type * job;
  job = util_malloc(sizeof * job , __func__);
  UTIL_TYPE_ID_INIT( job , LOCAL_JOB_TYPE_ID );
  job->active = false;
  job->status = JOB_QUEUE_WAITING;
  return job;
}

void local_job_free(local_job_type * job) {
  if (job->active) {
    /* Thread clean up */
  }
  free(job);
}



job_status_type local_driver_get_job_status(void * __driver, void * __job) {
  if (__job == NULL) 
    /* The job has not been registered at all ... */
    return JOB_QUEUE_NOT_ACTIVE;
  else {
    local_job_type    * job    = local_job_safe_cast( __job );
    {
      if (job->active == false) {
	util_abort("%s: internal error - should not query status on inactive jobs \n" , __func__);
	return JOB_QUEUE_NOT_ACTIVE; /* Dummy */
      } else 
	return job->status;
    }
  }
}



void local_driver_free_job(void * __driver , void * __job) {
  local_job_type    * job    = local_job_safe_cast( __job );
  local_job_free(job);
}



void local_driver_kill_job(void * __driver , void * __job) {
  local_job_type    * job    = local_job_safe_cast( __job );
  if (job->active)
    pthread_kill(job->run_thread , SIGABRT);
  local_job_free( job );
}




void * submit_job_thread__(void * __arg) {
  arg_pack_type * arg_pack = arg_pack_safe_cast(__arg);
  const char * executable  = arg_pack_iget_ptr(arg_pack , 0);
  const char * run_path    = arg_pack_iget_ptr(arg_pack , 1);
  local_job_type * job     = arg_pack_iget_ptr(arg_pack , 2);

  util_fork_exec(executable , 1 , &run_path , true , NULL , NULL , NULL , NULL , NULL); 
  job->status = JOB_QUEUE_DONE;
  pthread_exit(NULL);
  return NULL;
}



void * local_driver_submit_job(void * __driver, 
                               int   node_index                   , 
                               const char * submit_cmd  	  , 
                               const char * run_path    	  , 
                               const char * job_name              ,
                               const void * job_arg) {
  local_driver_type * driver = local_driver_safe_cast( __driver );
  {
    local_job_type * job    = local_job_alloc();
    arg_pack_type  * arg_pack = arg_pack_alloc();
    arg_pack_append_ptr( arg_pack , (char *) submit_cmd);
    arg_pack_append_ptr( arg_pack , (char *) run_path);
    arg_pack_append_ptr( arg_pack , job );
    pthread_mutex_lock( &driver->submit_lock );
    if (pthread_create( &job->run_thread , &driver->thread_attr , submit_job_thread__ , arg_pack) != 0) 
      util_abort("%s: failed to create run thread - aborting \n",__func__);
    
    job->active = true;
    job->status = JOB_QUEUE_RUNNING;
    pthread_mutex_unlock( &driver->submit_lock );
    
    return job;
  }
}



void local_driver_free(local_driver_type * driver) {
  pthread_attr_destroy ( &driver->thread_attr );
  free(driver);
  driver = NULL;
}


void local_driver_free__(void * __driver) {
  local_driver_type * driver = local_driver_safe_cast( __driver );
  local_driver_free( driver );
}


void * local_driver_alloc() {
  local_driver_type * local_driver = util_malloc(sizeof * local_driver , __func__);
  UTIL_TYPE_ID_INIT( local_driver , LOCAL_DRIVER_TYPE_ID);
  pthread_mutex_init( &local_driver->submit_lock , NULL );
  pthread_attr_init( &local_driver->thread_attr );
  pthread_attr_setdetachstate( &local_driver->thread_attr , PTHREAD_CREATE_DETACHED );
  
  local_driver->submit      	     = local_driver_submit_job;
  local_driver->get_status  	     = local_driver_get_job_status;
  local_driver->kill_job     	     = local_driver_kill_job;
  local_driver->free_job    	     = local_driver_free_job;
  local_driver->free_driver 	     = local_driver_free__;
  local_driver->display_info         = NULL;
  local_driver->driver_type          = LOCAL_DRIVER; 
  
  return local_driver;
}





#undef LOCAL_DRIVER_ID  
#undef LOCAL_JOB_ID    

/*****************************************************************/

