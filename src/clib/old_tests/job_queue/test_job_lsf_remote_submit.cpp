#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

#include <ert/util/test_util.hpp>
#include <ert/util/util.hpp>

#include <ert/job_queue/lsf_driver.hpp>

void test_submit(lsf_driver_type *driver, const char *server,
                 const char *bsub_cmd, const char *bjobs_cmd,
                 const char *bkill_cmd, const char *cmd) {

    test_assert_true(lsf_driver_set_option(driver, LSF_DEBUG_OUTPUT, "TRUE"));
    test_assert_true(lsf_driver_set_option(driver, LSF_SERVER, server));

    if (bsub_cmd != NULL)
        test_assert_true(lsf_driver_set_option(driver, LSF_BSUB_CMD, server));

    if (bjobs_cmd != NULL)
        test_assert_true(lsf_driver_set_option(driver, LSF_BJOBS_CMD, server));

    if (bkill_cmd != NULL)
        test_assert_true(lsf_driver_set_option(driver, LSF_BKILL_CMD, server));

    {
        char *run_path = util_alloc_cwd();
        auto *job = (lsf_job_type *)lsf_driver_submit_job(
            driver, cmd, 1, run_path, "NAME", 0, NULL);
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
            sleep(2);

            {
                int lsf_status = 0;
                for (int i = 0; i < 10; i++) {
                    lsf_status = lsf_driver_get_job_status_lsf(driver, job);
                    if (lsf_status != JOB_STAT_EXIT) {
                        sleep(2);
                    } else {
                        break;
                    }
                }
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
    util_install_signals();
    {
        int iarg;
        auto *driver = (lsf_driver_type *)lsf_driver_alloc();

        for (iarg = 2; iarg < argc; iarg++) {
            const char *server = argv[iarg];
            printf("Testing lsf server:%s \n", server);
            test_submit(driver, server, NULL, NULL, NULL, argv[1]);
        }

        lsf_driver_free(driver);
    }
    exit(0);
}
