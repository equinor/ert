#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <basic_queue_driver.h>
#include <lsf_driver.h>
#include <lsf/lsbatch.h>
#include <util.h>
#include <pthread.h>
#include <void_arg.h>

struct lsf_job_struct {
  int 	    __basic_id;
  int  	    __lsf_id;
  long int  lsf_jobnr;
};


struct lsf_driver_struct {
  BASIC_QUEUE_DRIVER_FIELDS
  int __lsf_id;
  char * resource_request;
  char * queue_name;
  struct submit      lsf_request;
  struct submitReply lsf_reply; 

  pthread_mutex_t    submit_lock;
};

/*****************************************************************/

#define LSF_DRIVER_ID  1001
#define LSF_JOB_ID     2001

void lsf_driver_assert_cast(const lsf_driver_type * queue_driver) {
  if (queue_driver->__lsf_id != LSF_DRIVER_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}

void lsf_driver_init(lsf_driver_type * queue_driver) {
  queue_driver->__lsf_id = LSF_DRIVER_ID;
}


void lsf_job_assert_cast(const lsf_job_type * queue_job) {
  if (queue_job->__lsf_id != LSF_JOB_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}



lsf_job_type * lsf_job_alloc() {
  lsf_job_type * job;
  job = util_malloc(sizeof * job , __func__);
  job->__lsf_id = LSF_JOB_ID;
  return job;
}

void lsf_job_free(lsf_job_type * job) {
  free(job);
}



#define CASE(S1,S2) case(S1):  status = S2; break;
ecl_job_status_type lsf_driver_get_job_status(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  if (__job == NULL) 
    /* The job has not been registered at all ... */
    return ecl_queue_null;
  else {
    lsf_job_type    * job    = (lsf_job_type    *) __job;
    lsf_driver_type * driver = (lsf_driver_type *) __driver;
    lsf_driver_assert_cast(driver); 
    lsf_job_assert_cast(job);
    {
      ecl_job_status_type status;
      struct jobInfoEnt *job_info;
      if (lsb_openjobinfo(job->lsf_jobnr , NULL , NULL , NULL , NULL , ALL_JOB) != 1) {
	fprintf(stderr,"%s: failed to get information about LSF job:%ld - aborting \n",__func__ , job->lsf_jobnr);
	abort();
      }
      job_info = lsb_readjobinfo( NULL );
      lsb_closejobinfo();

      switch (job_info->status) {
	CASE(JOB_STAT_PEND  , ecl_queue_pending);
	CASE(JOB_STAT_SSUSP , ecl_queue_running);
	CASE(JOB_STAT_RUN   , ecl_queue_running);
	CASE(JOB_STAT_EXIT  , ecl_queue_exit);
	CASE(JOB_STAT_DONE  , ecl_queue_done);
	CASE(JOB_STAT_PDONE , ecl_queue_done);
	CASE(JOB_STAT_PERR  , ecl_queue_exit);
	CASE(192            , ecl_queue_done); /* This 192 seems to pop up - where the fuck it comes frome  _PDONE + _USUSP ??? */
      default:
	fprintf(stderr,"%s: job:%ld lsf_status:%d not handled - aborting \n",__func__ , job->lsf_jobnr , job_info->status);
	status = ecl_queue_done; /* ????  */
      }
      
      return status;
    }
  }
}
#undef CASE



void lsf_driver_free_job(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  lsf_job_type    * job    = (lsf_job_type    *) __job;
  lsf_driver_type * driver = (lsf_driver_type *) __driver;
  lsf_driver_assert_cast(driver); 
  lsf_job_assert_cast(job);
  lsf_job_free(job);
}



void lsf_driver_abort_job(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  lsf_job_type    * job    = (lsf_job_type    *) __job;
  lsf_driver_type * driver = (lsf_driver_type *) __driver;
  lsf_driver_assert_cast(driver); 
  lsf_job_assert_cast(job);
  lsb_forcekilljob(job->lsf_jobnr);
  lsf_driver_free_job(__driver , __job);
}



basic_queue_job_type * lsf_driver_submit_job(basic_queue_driver_type * __driver, int queue_index , const char * submit_cmd , const char * run_path , const char * ecl_base , const char * ecl_version_id) {
  lsf_driver_type * driver = (lsf_driver_type *) __driver;
  lsf_driver_assert_cast(driver); 
  {
    lsf_job_type * job    = lsf_job_alloc();
    char * lsf_stdout = util_alloc_joined_string((const char *[4]) {run_path   , "/"      , ecl_base , ".LSF-stdout"}  , 4 , "");
    char * command    = util_alloc_joined_string((const char *[4]) {submit_cmd , run_path , ecl_base , ecl_version_id} , 4 , " ");
    
    pthread_mutex_lock( &driver->submit_lock );
    driver->lsf_request.jobName = (char *) ecl_base;
    driver->lsf_request.outFile = lsf_stdout;
    driver->lsf_request.command = command;
    job->lsf_jobnr = lsb_submit( &driver->lsf_request , &driver->lsf_reply );
    pthread_mutex_unlock( &driver->submit_lock );
    free(lsf_stdout);
    free(command);

    if (job->lsf_jobnr < 0) {
      fprintf(stderr,"%s: something failed when submitting job to lsf_system - aborting \n",__func__);
      abort();
    }
    
    {
      basic_queue_job_type * basic_job = (basic_queue_job_type *) job;
      basic_queue_job_init(basic_job);
      return basic_job;
    }
  }
}



void * lsf_driver_alloc(const char * queue_name , const char * resource_request) {
  lsf_driver_type * lsf_driver = util_malloc(sizeof * lsf_driver , __func__);
  lsf_driver->queue_name       = util_alloc_string_copy(queue_name);
  lsf_driver->resource_request = util_alloc_string_copy(resource_request);
  lsf_driver->__lsf_id         = LSF_DRIVER_ID;
  pthread_mutex_init( &lsf_driver->submit_lock , NULL );
  
  memset(&lsf_driver->lsf_request , 0 , sizeof (lsf_driver->lsf_request));
  lsf_driver->lsf_request.options   	   = SUB_QUEUE + SUB_RES_REQ + SUB_JOB_NAME + SUB_OUT_FILE;
  lsf_driver->lsf_request.queue     	   = lsf_driver->queue_name;
  lsf_driver->lsf_request.resReq    	   = lsf_driver->resource_request;
  lsf_driver->lsf_request.beginTime 	   = 0;
  lsf_driver->lsf_request.termTime   	   = 0;   
  lsf_driver->lsf_request.numProcessors    = 1;
  lsf_driver->lsf_request.maxNumProcessors = 1;
  {
    int i;
    for (i=0; i < LSF_RLIM_NLIMITS; i++) 
      lsf_driver->lsf_request.rLimits[i] = DEFAULT_RLIMIT;
  }
  lsf_driver->lsf_request.options2 = 0;
  
  lsf_driver->submit     = lsf_driver_submit_job;
  lsf_driver->get_status = lsf_driver_get_job_status;
  lsf_driver->abort_f    = lsf_driver_abort_job;
  lsf_driver->free_job   = lsf_driver_free_job;
  if (lsb_init(NULL) != 0) {
    fprintf(stderr,"%s failed to initialize LSF environment - aborting\n",__func__);
    abort();
  }
  {
    basic_queue_driver_type * basic_driver = (basic_queue_driver_type *) lsf_driver;
    basic_queue_driver_init(basic_driver);
    return basic_driver;
  }
}


void lsf_driver_free(lsf_driver_type * driver) {
  free(driver->resource_request);
  free(driver->queue_name);
  free(driver);
  driver = NULL;
}


#undef LSF_DRIVER_ID  
#undef LSF_JOB_ID    

/*****************************************************************/

