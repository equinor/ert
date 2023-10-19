#include <stdlib.h>
#include <unistd.h>

#include <vector>

#include <cassert>
#include <ert/util/test_util.hpp>
#include <ert/util/test_work_area.hpp>
#include <ert/util/util.hpp>

#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/slurm_driver.hpp>

void make_sleep_job(const char *fname, int sleep_time) {
    FILE *stream = fopen(fname, "w");
    assert(stream);
    fprintf(stream, "sleep %d \n", sleep_time);
    fclose(stream);

    mode_t fmode = S_IRWXU;
    chmod(fname, fmode);
}

void make_failed_job(const char *fname, int sleep_time) {
    FILE *stream = fopen(fname, "w");
    assert(stream);
    fprintf(stream, "sleep %d \n", sleep_time);
    fprintf(stream, "exit 1\n");
    fclose(stream);

    mode_t fmode = S_IRWXU;
    chmod(fname, fmode);
}

void run(double squeue_timeout) {
    ecl::util::TestArea ta("slurm_submit", true);
    queue_driver_type *driver = queue_driver_alloc(SLURM_DRIVER);
    std::vector<void *> jobs;
    const char *long_cmd = util_alloc_abs_path("long_run.sh");
    const char *ok_cmd = util_alloc_abs_path("ok_run.sh");
    const char *fail_cmd = util_alloc_abs_path("failed_run.sh");
    int num_jobs = 6;

    make_sleep_job(long_cmd, 10);
    make_sleep_job(ok_cmd, 1);
    make_failed_job(fail_cmd, 1);
    auto squeue_timeout_string = std::to_string(squeue_timeout);
    queue_driver_set_option(driver, SLURM_SQUEUE_TIMEOUT_OPTION,
                            squeue_timeout_string.c_str());

    for (int i = 0; i < num_jobs; i++) {
        std::string run_path = ta.test_cwd() + "/" + std::to_string(i);
        std::string job_name = "job" + std::to_string(i);
        util_make_path(run_path.c_str());
        if (i == 0)
            jobs.push_back(queue_driver_submit_job(
                driver, long_cmd, 1, run_path.c_str(), job_name.c_str()));
        else if (i == num_jobs - 1)
            jobs.push_back(queue_driver_submit_job(
                driver, fail_cmd, 1, run_path.c_str(), job_name.c_str()));
        else
            jobs.push_back(queue_driver_submit_job(
                driver, ok_cmd, 1, run_path.c_str(), job_name.c_str()));
    }

    while (true) {
        int active_count = 0;
        for (auto *job_ptr : jobs) {
            auto status = queue_driver_get_status(driver, job_ptr);
            if (status == JOB_QUEUE_RUNNING || status == JOB_QUEUE_PENDING ||
                status == JOB_QUEUE_WAITING)
                active_count += 1;
        }

        if (active_count == 0)
            break;

        auto *long_job = jobs[0];
        auto long_status = queue_driver_get_status(driver, long_job);
        if (long_status != JOB_QUEUE_IS_KILLED)
            queue_driver_kill_job(driver, long_job);

        usleep(100000);
    }

    for (int i = 0; i < num_jobs; i++) {
        auto *job_ptr = jobs[i];
        if (i == 0)
            test_assert_int_equal(queue_driver_get_status(driver, job_ptr),
                                  JOB_QUEUE_IS_KILLED);
        else if (i == num_jobs - 1)
            test_assert_int_equal(queue_driver_get_status(driver, job_ptr),
                                  JOB_QUEUE_EXIT);
        else
            test_assert_int_equal(queue_driver_get_status(driver, job_ptr),
                                  JOB_QUEUE_DONE);
    }

    for (auto *job_ptr : jobs)
        queue_driver_free_job(driver, job_ptr);

    queue_driver_free(driver);
}

int main(int argc, char **argv) {
    run(0);
    run(2);
    exit(0);
}
