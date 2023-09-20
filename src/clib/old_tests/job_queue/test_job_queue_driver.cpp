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

void get_driver_option_lists(job_driver_type driver_type,
                             std::vector<std::string> driver_options) {
    queue_driver_type *driver_ = queue_driver_alloc(driver_type);
    stringlist_type *option_list = stringlist_alloc_new();
    queue_driver_init_option_list(driver_, option_list);

    for (const auto &i : driver_options) {
        test_assert_true(stringlist_contains(option_list, i.c_str()));
    }

    stringlist_free(option_list);
    queue_driver_free(driver_);
}

void test_local_driver_no_get_set_options() {
    queue_driver_type *driver_local = queue_driver_alloc(LOCAL_DRIVER);
    stringlist_type *option_list = stringlist_alloc_new();
    queue_driver_init_option_list(driver_local, option_list);
    test_assert_util_abort(
        "local_driver_get_option",
        [](void *arg) {
            auto local_driver = static_cast<queue_driver_type *>(arg);
            queue_driver_get_option(local_driver, "NA");
        },
        driver_local);

    test_assert_util_abort(
        "local_driver_set_option",
        [](void *arg) {
            auto local_driver = static_cast<queue_driver_type *>(arg);
            queue_driver_set_option(local_driver, "NA", "NA");
        },
        driver_local);
    stringlist_free(option_list);
    queue_driver_free(driver_local);
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

    get_driver_option_lists(TORQUE_DRIVER, TORQUE_DRIVER_OPTIONS);
    get_driver_option_lists(SLURM_DRIVER, SLURM_DRIVER_OPTIONS);
    get_driver_option_lists(LSF_DRIVER, LSF_DRIVER_OPTIONS);

    test_local_driver_no_get_set_options();

    exit(0);
}
