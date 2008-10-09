#include <util.h>
#include <stdlib.h>
#include <string.h>
#include <lsf_request.h>
#include <ext_job.h>
#include <ext_joblist.h>
#include <set.h>
#include <stringlist.h>
#include <job_queue.h>

struct lsf_request_struct {
  char                   * request;        /* The current string representation of the complete request. */
  const ext_joblist_type * joblist;        /* Reference to the current list of installed jobs. The lsf_request instance does *NOT* own this. */
  const stringlist_type  * forward_model;  /* Reference to the current forward model - NOT owned by lsf-request. */
};


lsf_request_type * lsf_request_alloc(const ext_joblist_type * joblist) {
  lsf_request_type * lsf_request = util_malloc(sizeof * lsf_request , __func__);
  lsf_request->request = NULL;
  lsf_request->joblist = joblist;
  lsf_request->forward_model = NULL;
  return lsf_request;
}


void lsf_request_reset(lsf_request_type * lsf_request) {
  lsf_request->request = util_safe_free( lsf_request->request );
}



void lsf_request_free(lsf_request_type * lsf_request) {
  lsf_request_reset(lsf_request);
  free( lsf_request );
}




/**
   This function updates the internal lsf_request status, and also
   calls the job_queue with the new request.
*/

void  lsf_request_update(lsf_request_type * lsf_request , const stringlist_type * forward_model , job_queue_type * job_queue) {
  if (forward_model != lsf_request->forward_model) {
    lsf_request->forward_model = forward_model;
    lsf_request->requst = util_safe_free( lsf_request->request );
    /*
      Iterate forward model ....
    */
    for (int = 0; i < stringlist_get_size(forward_model); i++) {
      const char * resource_request;
      const ext_job_type * ext_job = ext_joblist_get_job( lsf_request->joblist , stringlist_iget(forward_model, i));
      resource_request = ext_job_get_lsf_resources( ext_job );
      if (resource_request != NULL) {
	
      }
    }
      
    job_queue_set_resource_request(job_queue , lsf_request->request);
  }
}




const char * lsf_request_get(const lsf_request_type * lsf_request) {
  return lsf_request->request;
}
