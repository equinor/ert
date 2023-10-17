#include "catch2/catch.hpp"
#include <string>

#include <ert/job_queue/lsf_driver.hpp>

std::string get_option(lsf_driver_type *driver, const char *option_key) {
    return std::string((const char *)lsf_driver_get_option(driver, option_key)
                           ?: "");
}

void test_option(lsf_driver_type *driver, const char *option,
                 const char *value) {
    REQUIRE(lsf_driver_set_option(driver, option, value));
    REQUIRE(get_option(driver, option) == value);
}

void test_status(int lsf_status, job_status_type job_status) {
    REQUIRE(lsf_driver_convert_status(lsf_status) == job_status);
}

TEST_CASE("job_lsf_test_options", "[job_lsf]") {
    auto *driver = (lsf_driver_type *)lsf_driver_alloc();
    REQUIRE_FALSE(lsf_driver_has_project_code(driver));

    // test setting values
    test_option(driver, LSF_BSUB_CMD, "Xbsub");
    test_option(driver, LSF_BJOBS_CMD, "Xbsub");
    test_option(driver, LSF_BKILL_CMD, "Xbsub");
    test_option(driver, LSF_LOGIN_SHELL, "shell");
    test_option(driver, LSF_BSUB_CMD, "bsub");
    test_option(driver, LSF_PROJECT_CODE, "my-ppu");
    test_option(driver, LSF_BJOBS_TIMEOUT, "1234");

    REQUIRE(lsf_driver_has_project_code(driver));

    // test unsetting/resetting options to default values
    test_option(driver, LSF_BSUB_CMD, "");
    test_option(driver, LSF_BJOBS_CMD, "");
    test_option(driver, LSF_BKILL_CMD, "");
    test_option(driver, LSF_RSH_CMD, "");
    test_option(driver, LSF_LOGIN_SHELL, "");
    test_option(driver, LSF_PROJECT_CODE, "");

    // Setting NULL to numerical options should leave the value unchanged
    lsf_driver_set_option(driver, LSF_BJOBS_TIMEOUT, NULL);
    REQUIRE(get_option(driver, LSF_BJOBS_TIMEOUT) == "1234");

    lsf_driver_free(driver);
}

TEST_CASE("job_lsf_test_tr", "[job_lsf]") {
    test_status(JOB_STAT_PEND, JOB_QUEUE_PENDING);
    test_status(JOB_STAT_PSUSP, JOB_QUEUE_RUNNING);
    test_status(JOB_STAT_USUSP, JOB_QUEUE_RUNNING);
    test_status(JOB_STAT_SSUSP, JOB_QUEUE_RUNNING);
    test_status(JOB_STAT_RUN, JOB_QUEUE_RUNNING);
    test_status(JOB_STAT_NULL, JOB_QUEUE_NOT_ACTIVE);
    test_status(JOB_STAT_DONE, JOB_QUEUE_DONE);
    test_status(JOB_STAT_EXIT, JOB_QUEUE_EXIT);
    test_status(JOB_STAT_UNKWN, JOB_QUEUE_UNKNOWN);
    test_status(JOB_STAT_DONE + JOB_STAT_PDONE, JOB_QUEUE_DONE);
}

TEST_CASE("job_lsf_test_submit_method", "[job_lsf]") {
    auto *driver = (lsf_driver_type *)lsf_driver_alloc();
    REQUIRE(lsf_driver_get_submit_method(driver) == LSF_SUBMIT_LOCAL_SHELL);
    lsf_driver_free(driver);
}
