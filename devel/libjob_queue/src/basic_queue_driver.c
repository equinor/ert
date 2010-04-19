#include <stdlib.h>
#include <stdio.h>
#include <basic_queue_driver.h>


				  


/*****************************************************************/

#define BASIC_JOB_ID     2000

void basic_queue_job_assert_cast(const basic_queue_job_type * queue_job) {
  if (queue_job->__id != BASIC_JOB_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}


void basic_queue_job_init(basic_queue_job_type * queue_job) {
  queue_job->__id = BASIC_JOB_ID;
}




#undef BASIC_JOB_ID    

/*****************************************************************/

