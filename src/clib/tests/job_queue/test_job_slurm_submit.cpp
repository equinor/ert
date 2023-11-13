#include "../tmpdir.hpp"
#include "catch2/catch.hpp"
#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/slurm_driver.hpp>
#include <filesystem>
#include <string>
#include <sys/stat.h>

void make_sleep_job(const char *fname, int sleep_time) {
    FILE *stream = fopen(fname, "w");
    REQUIRE(stream != nullptr);
    fprintf(stream, "sleep %d \n", sleep_time);
    fclose(stream);
    chmod(fname, S_IRWXU);
}

void submit_job(queue_driver_type *driver, bool expect_fail) {
    TmpDir tmpdir; // cwd is now a generated tmpdir
    std::string job_name = "JOB1";
    std::string cmd = tmpdir.get_current_tmpdir() + "/cmd.sh";
    make_sleep_job(cmd.c_str(), 10);
    std::string run_path = tmpdir.get_current_tmpdir() + "/" + job_name;
    std::filesystem::create_directory(std::filesystem::path(run_path));

    auto job = queue_driver_submit_job(driver, cmd, 1, run_path, job_name);
    if (expect_fail)
        REQUIRE(job == nullptr);
    else {
        REQUIRE_FALSE(job == nullptr);
        queue_driver_kill_job(driver, job);
        queue_driver_free_job(driver, job);
    }
}

TEST_CASE("job_slurm_submit_failure", "[job_slurm]") {
    queue_driver_type *driver = queue_driver_alloc(SLURM_DRIVER);
    queue_driver_set_option(driver, SLURM_PARTITION_OPTION,
                            "invalid_partition");
    submit_job(driver, true);
    queue_driver_free(driver);
}

TEST_CASE("job_slurm_submit_success", "[job_slurm]") {
    queue_driver_type *driver = queue_driver_alloc(SLURM_DRIVER);
    submit_job(driver, false);
    queue_driver_free(driver);
}
