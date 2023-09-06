#include <stdlib.h>

#include <ert/util/test_util.hpp>
#include <ert/util/util.hpp>

#include <ert/job_queue/job_queue.hpp>
#include <ert/job_queue/lsf_driver.hpp>

#include <ert/job_queue/slurm_driver.hpp>
#include <ert/job_queue/torque_driver.hpp>

void job_queue_set_driver_(job_driver_type driver_type) {
    job_queue_type *queue = job_queue_alloc();
    queue_driver_type *driver = queue_driver_alloc(driver_type);

    job_queue_set_driver(queue, driver);

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

void get_driver_option_lists() {
    //Torque driver option list
    {
        queue_driver_type *driver_torque = queue_driver_alloc(TORQUE_DRIVER);
        test_assert_string_equal(queue_driver_get_name(driver_torque),
                                 "TORQUE");
        stringlist_type *option_list = stringlist_alloc_new();
        queue_driver_init_option_list(driver_torque, option_list);

        for (const auto &i : TORQUE_DRIVER_OPTIONS) {
            test_assert_true(stringlist_contains(option_list, i.c_str()));
        }
        stringlist_free(option_list);
        queue_driver_free(driver_torque);
    }

    //Local driver option list (only general queue_driver options)
    {
        queue_driver_type *driver_local = queue_driver_alloc(LOCAL_DRIVER);
        test_assert_string_equal(queue_driver_get_name(driver_local), "local");
        stringlist_type *option_list = stringlist_alloc_new();
        queue_driver_init_option_list(driver_local, option_list);

        stringlist_free(option_list);
        queue_driver_free(driver_local);
    }

    //Lsf driver option list
    {
        queue_driver_type *driver_lsf = queue_driver_alloc(LSF_DRIVER);
        test_assert_string_equal(queue_driver_get_name(driver_lsf), "LSF");
        stringlist_type *option_list = stringlist_alloc_new();
        queue_driver_init_option_list(driver_lsf, option_list);

        for (const auto &i : LSF_DRIVER_OPTIONS) {
            test_assert_true(stringlist_contains(option_list, i.c_str()));
        }

        stringlist_free(option_list);
        queue_driver_free(driver_lsf);
    }

    //SLurm driver option list
    {
        queue_driver_type *driver_slurm = queue_driver_alloc(SLURM_DRIVER);
        test_assert_string_equal(queue_driver_get_name(driver_slurm), "SLURM");
        stringlist_type *option_list = stringlist_alloc_new();
        queue_driver_init_option_list(driver_slurm, option_list);

        for (const auto &i : SLURM_DRIVER_OPTIONS) {
            test_assert_true(stringlist_contains(option_list, i.c_str()));
        }

        stringlist_free(option_list);
        queue_driver_free(driver_slurm);
    }
}

int main(int argc, char **argv) {
    job_queue_set_driver_(LSF_DRIVER);
    job_queue_set_driver_(LOCAL_DRIVER);
    job_queue_set_driver_(TORQUE_DRIVER);
    job_queue_set_driver_(SLURM_DRIVER);

    set_option_invalid_option_returns_false();
    set_option_invalid_value_returns_false();

    set_option_valid_on_specific_driver_returns_true();
    get_driver_option_lists();

    exit(0);
}
