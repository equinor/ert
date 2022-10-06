#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

#include <ert/util/test_util.hpp>
#include <ert/util/util.hpp>

#include <ert/job_queue/lsf_driver.hpp>
#include <ert/job_queue/lsf_job_stat.hpp>

void test_submit(lsf_driver_type *driver) {
    test_assert_true(lsf_driver_set_option(driver, LSF_DEBUG_OUTPUT, "TRUE"));
    test_assert_int_equal(LSF_SUBMIT_INTERNAL,
                          lsf_driver_get_submit_method(driver));
    {
        char *run_path = util_alloc_cwd();
        lsf_job_type *job =
            lsf_driver_submit_job(driver, cmd, 1, run_path, "NAME", 0, NULL);
        if (job) {
            {
                int lsf_status = lsf_driver_get_job_status_lsf(driver, job);
                if (!((lsf_status == JOB_STAT_RUN) ||
                      (lsf_status == JOB_STAT_PEND)))
                    test_error_exit("Got lsf_status:%d expected: %d or %d \n",
                                    lsf_status, JOB_STAT_RUN, JOB_STAT_PEND);
            }

            lsf_driver_kill_job(driver, job);
            lsf_driver_set_bjobs_refresh_interval(driver, 0);
            sleep(1);

            {
                int lsf_status = lsf_driver_get_job_status_lsf(driver, job);
                if (lsf_status != JOB_STAT_EXIT)
                    test_error_exit("Got lsf_status:%d expected: %d \n",
                                    lsf_status, JOB_STAT_EXIT);
            }
        } else
            test_error_exit("lsf_driver_submit_job() returned NULL \n");

        free(run_path);
    }
}

int main(int argc, char **argv) {
    lsf_driver_type *driver = lsf_driver_alloc();
    test_submit(driver);
    lsf_driver_free(driver);
    exit(0);
}
