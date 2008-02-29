#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <ecl_queue.h>
#include <util.h>
#include <basic_queue_driver.h>


/**

All queue drivers must support the following functions:

  submit: This will submit a job, and return a pointer to a 
          newly allocated queue_job instance.

  clean:  This will clear up all resources used by the job.

  abort:  This will stop the job, and then call clean.

  status: This will get the status of the job.

*/ 



struct ecl_queue_struct {
  int size;
  basic_queue_driver_type * driver;
};



void ecl_queue_submit_job(ecl_queue_type * queue) {
  basic_queue_driver_type*driver = queue->driver;
  driver->submit(queue->driver , 1 , NULL , NULL , NULL , NULL);
}




ecl_queue_type * ecl_queue_alloc(int size , void * driver) {
  ecl_queue_type * queue = util_malloc(sizeof * queue , __func__);
  queue->size   = size;
  queue->driver = driver;
  basic_queue_driver_assert_cast(queue->driver);
  return queue;
}
