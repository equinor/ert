#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <ert/job_queue/torque_driver.hpp>
#include <ert/util/test_util.hpp>
#include <ert/util/test_work_area.hpp>

void test_option(torque_driver_type *driver, const char *option,
                 const char *value) {
    test_assert_true(torque_driver_set_option(driver, option, value));
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, option), value);
}

void setoption_setalloptions_optionsset() {
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

    char tmp_path[] = "/tmp/torque_debug_XXXXXX";
    // We do not strictly need the file, we are only interested in a path name
    // tmpnam is however deprecated in favor of mkstemp, and no good substitute
    // for tmpnam (with similar functionality) was found.
    int fd = mkstemp(tmp_path);

    if (fd == -1) {
        printf("Unable to create dummy log file");
        exit(1);
    }

    close(fd);
    unlink(tmp_path);

    test_option(driver, TORQUE_QSUB_CMD, NULL);
    test_option(driver, TORQUE_QSTAT_CMD, NULL);
    test_option(driver, TORQUE_QSTAT_OPTIONS, NULL);
    test_option(driver, TORQUE_QDEL_CMD, NULL);
    test_option(driver, TORQUE_QUEUE, NULL);
    test_option(driver, TORQUE_CLUSTER_LABEL, NULL);
    test_option(driver, TORQUE_JOB_PREFIX_KEY, NULL);

    // Setting NULL to numerical options should leave the value unchanged
    torque_driver_set_option(driver, TORQUE_NUM_CPUS_PER_NODE, NULL);
    torque_driver_set_option(driver, TORQUE_NUM_NODES, NULL);
    torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, NULL);
    torque_driver_set_option(driver, TORQUE_QUEUE_QUERY_TIMEOUT, NULL);
    test_assert_string_equal((const char *)torque_driver_get_option(
                                 driver, TORQUE_NUM_CPUS_PER_NODE),
                             "42");
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_NUM_NODES), "36");
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_KEEP_QSUB_OUTPUT),
        "0");
    test_assert_string_equal((const char *)torque_driver_get_option(
                                 driver, TORQUE_QUEUE_QUERY_TIMEOUT),
                             "128");

    torque_driver_free(driver);
}

void setoption_set_typed_options_wrong_format_returns_false() {
    torque_driver_type *driver = (torque_driver_type *)torque_driver_alloc();
    test_assert_false(
        torque_driver_set_option(driver, TORQUE_NUM_CPUS_PER_NODE, "42.2"));
    test_assert_false(
        torque_driver_set_option(driver, TORQUE_NUM_CPUS_PER_NODE, "fire"));
    test_assert_false(
        torque_driver_set_option(driver, TORQUE_NUM_NODES, "42.2"));
    test_assert_false(
        torque_driver_set_option(driver, TORQUE_NUM_NODES, "fire"));
    test_assert_true(
        torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "true"));
    test_assert_true(
        torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "1"));
    test_assert_false(
        torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "ja"));
    test_assert_false(
        torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "22"));
    test_assert_false(
        torque_driver_set_option(driver, TORQUE_KEEP_QSUB_OUTPUT, "1.1"));
    test_assert_false(
        torque_driver_set_option(driver, TORQUE_SUBMIT_SLEEP, "X45"));
    test_assert_false(
        torque_driver_set_option(driver, TORQUE_QUEUE_QUERY_TIMEOUT, "X45"));
}

void getoption_nooptionsset_defaultoptionsreturned() {
    torque_driver_type *driver = (torque_driver_type *)torque_driver_alloc();
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_QSUB_CMD),
        TORQUE_DEFAULT_QSUB_CMD);
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_QSTAT_CMD),
        TORQUE_DEFAULT_QSTAT_CMD);
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_QSTAT_OPTIONS),
        TORQUE_DEFAULT_QSTAT_OPTIONS);
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_QDEL_CMD),
        TORQUE_DEFAULT_QDEL_CMD);
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_KEEP_QSUB_OUTPUT),
        "0");
    test_assert_string_equal((const char *)torque_driver_get_option(
                                 driver, TORQUE_NUM_CPUS_PER_NODE),
                             "1");
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_NUM_NODES), "1");
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_CLUSTER_LABEL),
        NULL);
    test_assert_string_equal(
        (const char *)torque_driver_get_option(driver, TORQUE_JOB_PREFIX_KEY),
        NULL);

    printf("Default options OK\n");
    torque_driver_free(driver);
}

int main(int argc, char **argv) {
    getoption_nooptionsset_defaultoptionsreturned();
    setoption_setalloptions_optionsset();

    setoption_set_typed_options_wrong_format_returns_false();
    exit(0);
}
