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
#include <ert/util/thread_pool.h>
#include <ert/util/arg_pack.h>

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

void run_jobs_with_time_limit_test(char * executable_to_run, int number_of_jobs, int number_of_slowjobs, char * sleep_short, char * sleep_long, int max_sleep) {
  test_work_area_type * work_area = test_work_area_alloc("job_queue");
  
  job_queue_type * queue = job_queue_alloc(number_of_jobs, "OK.status", "ERROR");
  queue_driver_type * driver = queue_driver_alloc_local();
  job_queue_set_driver(queue, driver);
  job_queue_set_max_job_duration(queue, max_sleep);
  
  int submitted_slowjobs = 0;
  for (int i = 0; i < number_of_jobs; i++) {
    char * runpath = util_alloc_sprintf("%s/%s_%d", test_work_area_get_cwd(work_area), "job", i);
    util_make_path(runpath);

    char * sleeptime = sleep_short;
    if (submitted_slowjobs < number_of_slowjobs) {
      sleeptime = sleep_long;
      submitted_slowjobs++;
    }

    job_queue_add_job_st(queue, executable_to_run, NULL, NULL, NULL, NULL, 1, runpath, "Testjob", 2, (const char *[2]) { runpath, sleeptime });

    free(runpath);
  }

  job_queue_run_jobs(queue, number_of_jobs, true);

  test_assert_int_equal(number_of_jobs - number_of_slowjobs, job_queue_get_num_complete(queue));
  test_assert_int_equal(number_of_slowjobs, job_queue_get_num_killed(queue));
  
  test_assert_bool_equal(false, job_queue_get_open(queue));
  job_queue_reset(queue);
  test_assert_bool_equal(true, job_queue_get_open(queue));

  test_assert_int_equal(0, job_queue_get_num_complete(queue));

  job_queue_free(queue);
  queue_driver_free(driver);
  test_work_area_free(work_area);
}


void run_jobs_time_limit_multithreaded(char * executable_to_run, int number_of_jobs, int number_of_slowjobs, char * sleep_short, char * sleep_long, int max_sleep) {
  test_work_area_type * work_area = test_work_area_alloc("job_queue");
  
  job_queue_type * queue = job_queue_alloc(number_of_jobs, "OK.status", "ERROR");
  queue_driver_type * driver = queue_driver_alloc_local();
  job_queue_set_driver(queue, driver);
  job_queue_set_max_job_duration(queue, max_sleep);
  
  arg_pack_type * arg_pack = arg_pack_alloc();   
  arg_pack_append_ptr( arg_pack , queue );
  arg_pack_append_int( arg_pack , 0 );
  arg_pack_append_bool( arg_pack , true );
  
  thread_pool_type * pool = thread_pool_alloc(1, true);
  thread_pool_add_job(pool, job_queue_run_jobs__, arg_pack);
  
  int submitted_slowjobs = 0;
  for (int i = 0; i < number_of_jobs; i++) {
    char * runpath = util_alloc_sprintf("%s/%s_%d", test_work_area_get_cwd(work_area), "job", i);
    util_make_path(runpath);

    char * sleeptime = sleep_short;
    if (submitted_slowjobs < number_of_slowjobs) {
      sleeptime = sleep_long;
      submitted_slowjobs++;
    }

    job_queue_add_job_mt(queue, executable_to_run, NULL, NULL, NULL, NULL, 1, runpath, "Testjob", 2, (const char *[2]) { runpath, sleeptime });

    free(runpath);
  }
  
  job_queue_submit_complete(queue);
  thread_pool_join(pool);

  test_assert_int_equal(number_of_jobs - number_of_slowjobs, job_queue_get_num_complete(queue));
  test_assert_int_equal(number_of_slowjobs, job_queue_get_num_killed(queue));
  test_assert_bool_equal(false, job_queue_get_open(queue));
  job_queue_reset(queue);
  test_assert_bool_equal(true, job_queue_get_open(queue));
  test_assert_int_equal(0, job_queue_get_num_complete(queue));
  
  job_queue_free(queue);
  queue_driver_free(driver);
  test_work_area_free(work_area);
}


int main(int argc, char ** argv) {
  job_queue_set_driver_(LSF_DRIVER);
  job_queue_set_driver_(TORQUE_DRIVER);
  run_jobs_with_time_limit_test(argv[1], 10, 0, "1", "100", 0); // 0 as limit means no limit
  run_jobs_with_time_limit_test(argv[1], 100, 23, "1", "100", 5);
  
  run_jobs_time_limit_multithreaded(argv[1], 100, 23, "1", "100", 5);
  exit(0);
}
