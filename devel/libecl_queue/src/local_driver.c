#include <stdlib.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <basic_queue_driver.h>
#include <local_driver.h>
#include <util.h>
#include <pthread.h>
#include <void_arg.h>

struct local_job_struct {
  int 	     __basic_id;
  int  	     __local_id;
  bool       active;
  pthread_t  run_thread;
};



struct local_driver_struct {
  BASIC_QUEUE_DRIVER_FIELDS
  int __local_id;
  pthread_attr_t     thread_attr;
  pthread_mutex_t    submit_lock;
};

/*****************************************************************/

#define LOCAL_DRIVER_ID  1002
#define LOCAL_JOB_ID     2002

void local_driver_assert_cast(const local_driver_type * queue_driver) {
  if (queue_driver->__local_id != LOCAL_DRIVER_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}

void local_driver_init(local_driver_type * queue_driver) {
  queue_driver->__local_id = LOCAL_DRIVER_ID;
}


void local_job_assert_cast(const local_job_type * queue_job) {
  if (queue_job->__local_id != LOCAL_JOB_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}



local_job_type * local_job_alloc() {
  local_job_type * job;
  job = util_malloc(sizeof * job , __func__);
  job->__local_id = LOCAL_JOB_ID;
  job->active = false;
  return job;
}

void local_job_free(local_job_type * job) {
  if (job->active) {
    /* Thread clean up */
  }
  free(job);
}



ecl_job_status_type local_driver_get_job_status(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  if (__job == NULL) 
    /* The job has not been registered at all ... */
    return ecl_queue_null;
  else {
    local_job_type    * job    = (local_job_type    *) __job;
    local_driver_type * driver = (local_driver_type *) __driver;
    local_driver_assert_cast(driver); 
    local_job_assert_cast(job);
    {
      ecl_job_status_type status;
      if (job->active == false) {
	fprintf(stderr,"%s: internal error - should not query status on inactive jobs \n" , __func__);
	abort();
      } else {
	if (pthread_kill(job->run_thread , 0) == 0)
	  status = ecl_queue_running;
	else
	  status = ecl_queue_done;
      }
      return status;
    }
  }
}



void local_driver_free_job(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  local_job_type    * job    = (local_job_type    *) __job;
  local_driver_type * driver = (local_driver_type *) __driver;
  local_driver_assert_cast(driver); 
  local_job_assert_cast(job);
  local_job_free(job);
}



void local_driver_abort_job(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  local_job_type    * job    = (local_job_type    *) __job;
  local_driver_type * driver = (local_driver_type *) __driver;
  local_driver_assert_cast(driver); 
  local_job_assert_cast(job);
  if (job->active)
    pthread_kill(job->run_thread , SIGABRT);
  local_driver_free_job(__driver , __job);
}


void * submit_job_thread(void * cmd) {
  system((const char *) cmd);
  free(cmd);
  pthread_exit(NULL);
  return NULL;
}


basic_queue_job_type * local_driver_submit_job(basic_queue_driver_type * __driver, 
					       int   node_index                   , 
					       const char * submit_cmd  	  , 
					       const char * run_path    	  , 
					       const char * ecl_base    	  , 
					       const char * eclipse_exe 	  ,   
					       const char * eclipse_config  , 
					       const char * eclipse_LD_path , 
					       const char * license_server) {
  local_driver_type * driver = (local_driver_type *) __driver;
  local_driver_assert_cast(driver); 
  {
    local_job_type * job    = local_job_alloc();
    /* 
       command is freed in the start_routine() function
    */
    char * command;
    if (eclipse_LD_path == NULL)
      command = util_alloc_joined_string((const char*[6]) {submit_cmd , run_path , ecl_base , eclipse_exe , eclipse_config , license_server} , 6 , " ");
    else
      command = util_alloc_joined_string((const char*[7]) {submit_cmd , run_path , ecl_base , eclipse_exe , eclipse_config , license_server , eclipse_LD_path} , 7 , " ");
    
    pthread_mutex_lock( &driver->submit_lock );
    if (pthread_create( &job->run_thread , &driver->thread_attr , submit_job_thread , command) != 0) {
      fprintf(stderr,"%s: failed to create run thread - aborting \n",__func__);
      abort();
    }
    job->active = true;
    pthread_mutex_unlock( &driver->submit_lock );
    
    {
      basic_queue_job_type * basic_job = (basic_queue_job_type *) job;
      basic_queue_job_init(basic_job);
      return basic_job;
    }
  }
}



void * local_driver_alloc() {
  local_driver_type * local_driver = util_malloc(sizeof * local_driver , __func__);
  local_driver->__local_id         = LOCAL_DRIVER_ID;
  pthread_mutex_init( &local_driver->submit_lock , NULL );
  pthread_attr_init( &local_driver->thread_attr );
  pthread_attr_setdetachstate( &local_driver->thread_attr , PTHREAD_CREATE_DETACHED );
  
  local_driver->submit      = local_driver_submit_job;
  local_driver->get_status  = local_driver_get_job_status;
  local_driver->abort_f     = local_driver_abort_job;
  local_driver->free_job    = local_driver_free_job;
  local_driver->free_driver = local_driver_free__;
  {
    basic_queue_driver_type * basic_driver = (basic_queue_driver_type *) local_driver;
    basic_queue_driver_init(basic_driver);
    return basic_driver;
  }
}


void local_driver_free(local_driver_type * driver) {
  pthread_attr_destroy ( &driver->thread_attr );
  free(driver);
  driver = NULL;
}


void local_driver_free__(basic_queue_driver_type * driver) {
  local_driver_free((local_driver_type *) driver);
}



#undef LOCAL_DRIVER_ID  
#undef LOCAL_JOB_ID    

/*****************************************************************/

