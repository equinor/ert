/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'job_queue_test.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <stdlib.h>
#include <stdbool.h>

#include <ert/util/util.h>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>

#include <ert/job_queue/job_queue.h>

void job_queue_set_driver_(job_driver_type driver_type) {
  job_queue_type * queue = job_queue_alloc(10, "OK", "ERROR");
  queue_driver_type * driver = queue_driver_alloc(driver_type);
  test_assert_false(job_queue_has_driver(queue));

  job_queue_set_driver(queue, driver);
  test_assert_true(job_queue_has_driver(queue));

  job_queue_free(queue);
  queue_driver_free(driver);

}

void basic_run_jobs_test(char * executable_to_run, int number_of_jobs) {
  job_queue_type * queue = job_queue_alloc(number_of_jobs, "OK.status", "ERROR");
  queue_driver_type * driver = queue_driver_alloc_local();
  job_queue_set_driver(queue, driver);

  test_work_area_type * work_area = test_work_area_alloc("job_queue", false);

  for (int i = 0; i < number_of_jobs; i++) {
    char * runpath = util_alloc_sprintf("%s/%s_%d", test_work_area_get_cwd(work_area), "job", i);
    util_make_path(runpath);
    char * sleep_time = "1";

    job_queue_add_job_st(queue, executable_to_run, NULL, NULL, NULL, NULL, 1, runpath, "Testjob", 2, (const char *[2]) {
      runpath, sleep_time
    });
    free(runpath);
  }
  job_queue_run_jobs_finalizeoptional(queue, number_of_jobs, true, false);

  test_assert_int_equal(number_of_jobs, job_queue_get_num_complete(queue));
  job_queue_finalize(queue);
  test_assert_int_equal(0, job_queue_get_num_complete(queue));

  job_queue_free(queue);
  queue_driver_free(driver);
  test_work_area_free(work_area);
}

void run_jobs_with_time_limit_test(char * executable_to_run, int number_of_jobs, int number_of_slowjobs) {
  test_work_area_type * work_area = test_work_area_alloc("job_queue", false);
  
  job_queue_type * queue = job_queue_alloc(number_of_jobs, "OK.status", "ERROR");
  queue_driver_type * driver = queue_driver_alloc_local();
  job_queue_set_driver(queue, driver);
  job_queue_set_max_job_duration(queue, 5);
  
  for (int i = 0; i < number_of_jobs; i++) {
    char * runpath = util_alloc_sprintf("%s/%s_%d", test_work_area_get_cwd(work_area), "job", i);
    util_make_path(runpath);

    char * sleeptime = "1";
    if (number_of_slowjobs-- > 0)
      sleeptime = "10000";

    job_queue_add_job_st(queue, executable_to_run, NULL, NULL, NULL, NULL, 1, runpath, "Testjob", 2, (const char *[2]) { runpath, sleeptime });

    free(runpath);
  }

  job_queue_run_jobs_finalizeoptional(queue, number_of_jobs, true, false);

  test_assert_int_equal(number_of_jobs, job_queue_get_num_complete(queue));
  job_queue_finalize(queue);
  test_assert_int_equal(0, job_queue_get_num_complete(queue));

  job_queue_free(queue);
  queue_driver_free(driver);
  test_work_area_free(work_area);
}

int main(int argc, char ** argv) {
  job_queue_set_driver_(LSF_DRIVER);
  //job_queue_set_driver_(TORQUE_DRIVER);
  //basic_run_jobs_test(argv[1], 100);

  run_jobs_with_time_limit_test(argv[1], 10, 2);

  exit(0);
}

/*
 int job_queue_add_job_st(job_queue_type * queue , 
                         const char * run_cmd , 
                         job_callback_ftype * done_callback, 
                         job_callback_ftype * retry_callback, 
                         void * callback_arg , 
                         int num_cpu , 
                         const char * run_path , 
                         const char * job_name , 
                         int argc , 
             
 *             const char ** argv) {
 */

