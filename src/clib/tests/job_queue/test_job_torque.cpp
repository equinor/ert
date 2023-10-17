#include "catch2/catch.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include <ert/job_queue/torque_driver.hpp>

std::string get_option(torque_driver_type *driver, const char *option_key) {
    return std::string(
        (const char *)torque_driver_get_option(driver, option_key) ?: "");
}

void test_option(torque_driver_type *driver, const char *option,
                 const char *value) {
    REQUIRE(torque_driver_set_option(driver, option, value));
    REQUIRE(get_option(driver, option) == std::string(value));
}

TEST_CASE("job_torque_set_all_options_set", "[job_torque]") {
    torque_driver_type *driver = (torque_driver_type *)torque_driver_alloc();

    test_option(driver, TORQUE_QSUB_CMD, "XYZaaa");
    test_option(driver, TORQUE_QSTAT_CMD, "xyZfff");
    test_option(driver, TORQUE_QSTAT_OPTIONS, "-magic");
    test_option(driver, TORQUE_QDEL_CMD, "ZZyfff");
    test_option(driver, TORQUE_QUEUE, "superhigh");
    test_option(driver, TORQUE_NUM_CPUS_PER_NODE, "42");
    test_option(driver, TORQUE_NUM_NODES, "36");
    test_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "1");
    test_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "0");
    test_option(driver, TORQUE_CLUSTER_LABEL, "thecluster");
    test_option(driver, TORQUE_JOB_PREFIX_KEY, "coolJob");
    test_option(driver, TORQUE_QUEUE_QUERY_TIMEOUT, "128");

    test_option(driver, TORQUE_QSUB_CMD, "");
    test_option(driver, TORQUE_QSTAT_CMD, "");
    test_option(driver, TORQUE_QSTAT_OPTIONS, "");
    test_option(driver, TORQUE_QDEL_CMD, "");
    test_option(driver, TORQUE_QUEUE, "");
    test_option(driver, TORQUE_CLUSTER_LABEL, "");
    test_option(driver, TORQUE_JOB_PREFIX_KEY, "");

    // Setting NULL to numerical options should leave the value unchanged
    torque_driver_set_option(driver, TORQUE_NUM_CPUS_PER_NODE, NULL);
    torque_driver_set_option(driver, TORQUE_NUM_NODES, NULL);
    torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, NULL);
    torque_driver_set_option(driver, TORQUE_QUEUE_QUERY_TIMEOUT, NULL);

    REQUIRE(get_option(driver, TORQUE_NUM_CPUS_PER_NODE) == "42");
    REQUIRE(get_option(driver, TORQUE_NUM_NODES) == "36");
    REQUIRE(get_option(driver, TORQUE_KEEP_QSUB_OUTPUT) == "0");
    REQUIRE(get_option(driver, TORQUE_QUEUE_QUERY_TIMEOUT) == "128");
    torque_driver_free(driver);
}

TEST_CASE("job_torque_set_typed_options_wrong_format_returns_false",
          "[job_torque]") {
    torque_driver_type *driver = (torque_driver_type *)torque_driver_alloc();
    REQUIRE_FALSE(
        torque_driver_set_option(driver, TORQUE_NUM_CPUS_PER_NODE, "42.2"));
    REQUIRE_FALSE(
        torque_driver_set_option(driver, TORQUE_NUM_CPUS_PER_NODE, "fire"));
    REQUIRE_FALSE(torque_driver_set_option(driver, TORQUE_NUM_NODES, "42.2"));
    REQUIRE_FALSE(torque_driver_set_option(driver, TORQUE_NUM_NODES, "fire"));
    REQUIRE(torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "true"));
    REQUIRE(torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "1"));
    REQUIRE_FALSE(
        torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "ja"));
    REQUIRE_FALSE(
        torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "22"));
    REQUIRE_FALSE(
        torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "1.1"));
    REQUIRE_FALSE(torque_driver_set_option(driver, TORQUE_SUBMIT_SLEEP, "X45"));
    REQUIRE_FALSE(
        torque_driver_set_option(driver, TORQUE_QUEUE_QUERY_TIMEOUT, "X45"));
    torque_driver_free(driver);
}

TEST_CASE("job_torque_get_option_no_option_set_default_options_returned",
          "[job_torque]") {
    torque_driver_type *driver = (torque_driver_type *)torque_driver_alloc();
    REQUIRE(get_option(driver, TORQUE_QSUB_CMD) == TORQUE_DEFAULT_QSUB_CMD);
    REQUIRE(get_option(driver, TORQUE_QSTAT_CMD) == TORQUE_DEFAULT_QSTAT_CMD);
    REQUIRE(get_option(driver, TORQUE_QSTAT_OPTIONS) ==
            TORQUE_DEFAULT_QSTAT_OPTIONS);
    REQUIRE(get_option(driver, TORQUE_QDEL_CMD) == TORQUE_DEFAULT_QDEL_CMD);
    REQUIRE(get_option(driver, TORQUE_KEEP_QSUB_OUTPUT) == "0");
    REQUIRE(get_option(driver, TORQUE_NUM_CPUS_PER_NODE) == "1");
    REQUIRE(get_option(driver, TORQUE_NUM_NODES) == "1");
    REQUIRE(get_option(driver, TORQUE_CLUSTER_LABEL) == "");
    REQUIRE(get_option(driver, TORQUE_JOB_PREFIX_KEY) == "");
    torque_driver_free(driver);
}
