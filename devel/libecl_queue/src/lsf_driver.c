#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <basic_queue_driver.h>
#include <lsf_driver.h>
#include <lsf/lsbatch.h>
#include <util.h>

struct lsf_job_struct {
  int 	    __basic_id;
  int  	    __lsf_id;
  long int  lsf_jobnr;
};


struct lsf_driver_struct {
  int __basic_id;
  int __lsf_id;
  char * resource_request;
  char * queue_name;
  struct submit lsf_request;

  submit_job_ftype * submit;
  clean_job_ftype  * clean;
  abort_job_ftype  * abort_f;
  get_status_ftype * get_status;
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
  if (queue_job->__lsf_id != LSF_DRIVER_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}


void lsf_job_init(lsf_job_type * queue_job) {
  queue_job->__lsf_id = LSF_DRIVER_ID;
}




basic_queue_job_type * lsf_driver_submit_job(basic_queue_driver_type * driver, int queue_index) {
  lsf_job_type * job = util_malloc(sizeof * job , __func__);
  job->__lsf_id = LSF_JOB_ID;
  {
    basic_queue_job_type * basic_job = (basic_queue_job_type *) job;
    basic_queue_job_init(basic_job);
    return basic_job;
  }
}



basic_queue_driver_type * lsf_driver_alloc(const char * queue_name , const char * resource_request) {
  lsf_driver_type * lsf_driver = util_malloc(sizeof * lsf_driver , __func__);
  lsf_driver->queue_name       = util_alloc_string_copy(queue_name);
  lsf_driver->resource_request = util_alloc_string_copy(resource_request);
  lsf_driver->__lsf_id         = LSF_DRIVER_ID;

  
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
  

  {
    basic_queue_driver_type * basic_driver = (basic_queue_driver_type *) lsf_driver;
    basic_queue_driver_init(basic_driver);
    return basic_driver;
  }
}



#undef LSF_DRIVER_ID  
#undef LSF_JOB_ID    

/*****************************************************************/

