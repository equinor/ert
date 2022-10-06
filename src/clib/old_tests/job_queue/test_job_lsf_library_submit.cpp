#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

#include <assert.h>
#include <ert/util/util.hpp>

#include <ert/job_queue/lsf_driver.hpp>
#include <ert/job_queue/lsf_job_stat.hpp>

void test_submit(lsf_driver_type *driver, const char *cmd) {
    assert(lsf_driver_set_option(driver, LSF_DEBUG_OUTPUT, "TRUE"));
    assert(LSF_SUBMIT_INTERNAL == lsf_driver_get_submit_method(driver));
    {
        char *run_path = util_alloc_cwd();
        lsf_job_type *job =
            lsf_driver_submit_job(driver, cmd, 1, run_path, "NAME", 0, NULL);
        assert(job);
        {
            {
                int lsf_status = lsf_driver_get_job_status_lsf(driver, job);
                assert((lsf_status == JOB_STAT_RUN) ||
                       (lsf_status == JOB_STAT_PEND));
            }

            lsf_driver_kill_job(driver, job);
            lsf_driver_set_bjobs_refresh_interval(driver, 0);
            sleep(1);

            {
                int lsf_status = lsf_driver_get_job_status_lsf(driver, job);
                assert(lsf_status == JOB_STAT_EXIT);
            }
        }

        free(run_path);
    }
}

int main(int argc, char **argv) {
    lsf_driver_type *driver = lsf_driver_alloc();
    test_submit(driver, argv[1]);
    lsf_driver_free(driver);
    exit(0);
}
