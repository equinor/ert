#include <ert/util/test_util.hpp>
#include <ert/util/util.hpp>
#include <stdlib.h>
#include <vector>

#include <ert/job_queue/job_queue.hpp>
#include <ert/job_queue/local_driver.hpp>
#include <ert/job_queue/lsf_driver.hpp>
#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/slurm_driver.hpp>
#include <ert/job_queue/torque_driver.hpp>

void job_queue_set_driver_(job_driver_type driver_type) {
    queue_driver_type *driver = queue_driver_alloc(driver_type);
    job_queue_type *queue = job_queue_alloc(driver);
    job_queue_free(queue);
    queue_driver_free(driver);
}

void set_option_invalid_option_returns_false() {
    queue_driver_type *driver_torque = queue_driver_alloc(TORQUE_DRIVER);
    test_assert_false(
        queue_driver_set_option(driver_torque, "MAKS_RUNNING", "42"));
    queue_driver_free(driver_torque);
}

void set_option_invalid_value_returns_false() {
    queue_driver_type *driver_torque = queue_driver_alloc(TORQUE_DRIVER);
    test_assert_false(
        queue_driver_set_option(driver_torque, TORQUE_NUM_CPUS_PER_NODE, "2a"));
    queue_driver_free(driver_torque);
}

void set_option_valid_on_specific_driver_returns_true() {
    queue_driver_type *driver_torque = queue_driver_alloc(TORQUE_DRIVER);
    test_assert_true(
        queue_driver_set_option(driver_torque, TORQUE_NUM_CPUS_PER_NODE, "33"));
    test_assert_string_equal(
        "33", (const char *)queue_driver_get_option(driver_torque,
                                                    TORQUE_NUM_CPUS_PER_NODE));
    queue_driver_free(driver_torque);
}

int main(int argc, char **argv) {
    util_install_signals();
    job_queue_set_driver_(LSF_DRIVER);
    job_queue_set_driver_(LOCAL_DRIVER);
    job_queue_set_driver_(TORQUE_DRIVER);
    job_queue_set_driver_(SLURM_DRIVER);

    set_option_invalid_option_returns_false();
    set_option_invalid_value_returns_false();

    set_option_valid_on_specific_driver_returns_true();

    exit(0);
}
