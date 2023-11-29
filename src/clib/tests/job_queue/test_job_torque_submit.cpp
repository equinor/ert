#include "catch2/catch.hpp"
#include <ert/job_queue/torque_driver.hpp>
#include <filesystem>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

void test_submit(torque_driver_type *driver, const char *cmd) {
    std::string run_path = std::filesystem::current_path().string();

    auto *job = (torque_job_type *)torque_driver_submit_job(
        driver, cmd, 1, run_path.c_str(), "TEST-TORQUE-SUBMIT");

    REQUIRE(job != nullptr);
    REQUIRE((torque_driver_get_job_status(driver, job) &
             (JOB_QUEUE_RUNNING + JOB_QUEUE_PENDING)) != 0);
    torque_driver_kill_job(driver, job);

    printf("Waiting 3 seconds");
    for (int i = 0; i < 3; i++) {
        printf(".");
        fflush(stdout);
        sleep(1);
    }
    printf("\n");

    int torque_status = torque_driver_get_job_status(driver, job);

    REQUIRE((torque_status & (JOB_QUEUE_EXIT + JOB_QUEUE_DONE)) != 0);

    torque_driver_free_job(job);
}

void test_submit_failed_qstat(torque_driver_type *driver, const char *cmd) {
    std::string run_path = std::filesystem::current_path().string();

    torque_job_type *job = (torque_job_type *)torque_driver_submit_job(
        driver, cmd, 1, run_path.c_str(), "TEST-TORQUE-SUBMIT");

    std::string qstat_cmd = run_path + "/qstat.local";

    FILE *stream = fopen(qstat_cmd.c_str(), "w");
    REQUIRE(stream != nullptr);
    fprintf(stream, "#!/bin/sh\n");
    fprintf(stream, "echo XYZ - Error\n");
    fclose(stream);
    chmod(qstat_cmd.c_str(), S_IXUSR);

    torque_driver_set_option(driver, TORQUE_QSTAT_CMD, qstat_cmd.c_str());

    REQUIRE((torque_driver_get_job_status(driver, job) &
             JOB_QUEUE_STATUS_FAILURE) != 0);
    torque_driver_free_job(job);
}

TEST_CASE("job_torque_submit", "[job_torque]") {
    auto *driver = (torque_driver_type *)torque_driver_alloc();
    test_submit(driver, "dummy");
    torque_driver_free(driver);
}

TEST_CASE("job_torque_submit_failed_qstat", "[job_torque]") {
    auto *driver = (torque_driver_type *)torque_driver_alloc();
    test_submit_failed_qstat(driver, "dummy");
    torque_driver_free(driver);
}
