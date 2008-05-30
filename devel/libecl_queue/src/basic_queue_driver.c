#include <stdlib.h>
#include <stdio.h>
#include <basic_queue_driver.h>


				  


/*****************************************************************/

#define BASIC_DRIVER_ID  1000
#define BASIC_JOB_ID     2000

void basic_queue_driver_assert_cast(const basic_queue_driver_type * queue_driver) {
  if (queue_driver->__id != BASIC_DRIVER_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}

void basic_queue_driver_init(basic_queue_driver_type * queue_driver) {
  queue_driver->__id = BASIC_DRIVER_ID;
}

void basic_queue_job_assert_cast(const basic_queue_job_type * queue_job) {
  if (queue_job->__id != BASIC_DRIVER_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}

void basic_queue_job_init(basic_queue_job_type * queue_job) {
  queue_job->__id = BASIC_DRIVER_ID;
}

#undef BASIC_DRIVER_ID  
#undef BASIC_JOB_ID    

/*****************************************************************/

