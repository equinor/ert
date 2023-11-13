#include "../tmpdir.hpp"
#include "catch2/catch.hpp"
#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/slurm_driver.hpp>
#include <filesystem>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

void make_sleep_job(const char *fname, int sleep_time) {
    FILE *stream = fopen(fname, "w");
    REQUIRE(stream != nullptr);
    fprintf(stream, "sleep %d \n", sleep_time);
    fclose(stream);
    chmod(fname, S_IRWXU);
}

void make_failed_job(const char *fname, int sleep_time) {
    FILE *stream = fopen(fname, "w");
    REQUIRE(stream != nullptr);
    fprintf(stream, "sleep %d \n", sleep_time);
    fprintf(stream, "exit 1\n");
    fclose(stream);
    chmod(fname, S_IRWXU);
}

void run(double squeue_timeout) {
    TmpDir tmpdir; // cwd is now a generated tmpdir
    queue_driver_type *driver = queue_driver_alloc(SLURM_DRIVER);
    std::vector<void *> jobs;
    std::string long_str = tmpdir.get_current_tmpdir() + "/long_run.sh";
    std::string ok_str = tmpdir.get_current_tmpdir() + "/ok_run.sh";
    std::string fail_str = tmpdir.get_current_tmpdir() + "/fail_run.sh";
    const char *long_cmd = long_str.c_str();
    const char *ok_cmd = ok_str.c_str();
    const char *fail_cmd = fail_str.c_str();
    int num_jobs = 6;

    make_sleep_job(long_cmd, 10);
    make_sleep_job(ok_cmd, 1);
    make_failed_job(fail_cmd, 1);
    auto squeue_timeout_string = std::to_string(squeue_timeout);
    queue_driver_set_option(driver, SLURM_SQUEUE_TIMEOUT_OPTION,
                            squeue_timeout_string.c_str());

    for (int i = 0; i < num_jobs; i++) {
        std::string run_path =
            tmpdir.get_current_tmpdir() + "/" + std::to_string(i);
        std::string job_name = "job" + std::to_string(i);

        std::filesystem::create_directory(std::filesystem::path(run_path));

        if (i == 0)
            jobs.push_back(queue_driver_submit_job(driver, long_cmd, 1,
                                                   run_path, job_name));
        else if (i == num_jobs - 1)
            jobs.push_back(queue_driver_submit_job(driver, fail_cmd, 1,
                                                   run_path, job_name));
        else
            jobs.push_back(
                queue_driver_submit_job(driver, ok_cmd, 1, run_path, job_name));
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
        job_status_type job_status = JOB_QUEUE_DONE;
        if (i == 0)
            job_status = JOB_QUEUE_IS_KILLED;
        else if (i == num_jobs - 1)
            job_status = JOB_QUEUE_EXIT;

        REQUIRE(queue_driver_get_status(driver, job_ptr) == job_status);
    }

    for (auto *job_ptr : jobs)
        queue_driver_free_job(driver, job_ptr);

    queue_driver_free(driver);
}

TEST_CASE("job_slurm_runtest_timeout_0", "[job_slurm]") { run(0); }
TEST_CASE("job_slurm_runtest_timeout_2", "[job_slurm]") { run(2); }
