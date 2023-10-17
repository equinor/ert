#include "catch2/catch.hpp"
#include <cstdlib>
#include <string>

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

TEST_CASE("set_option_invalid_option_returns_false", "[job_queue]") {
    queue_driver_type *driver_torque = queue_driver_alloc(TORQUE_DRIVER);
    REQUIRE_FALSE(queue_driver_set_option(driver_torque, "MAKS_RUNNING", "42"));
    queue_driver_free(driver_torque);
}

TEST_CASE("set_option_invalid_value_returns_false", "[job_queue]") {
    queue_driver_type *driver_torque = queue_driver_alloc(TORQUE_DRIVER);
    REQUIRE_FALSE(
        queue_driver_set_option(driver_torque, TORQUE_NUM_CPUS_PER_NODE, "2a"));
    queue_driver_free(driver_torque);
}

TEST_CASE("set_option_valid_on_specific_driver_returns_true", "[job_queue]") {
    queue_driver_type *driver_torque = queue_driver_alloc(TORQUE_DRIVER);
    REQUIRE(
        queue_driver_set_option(driver_torque, TORQUE_NUM_CPUS_PER_NODE, "33"));
    REQUIRE("33" == std::string((const char *)queue_driver_get_option(
                        driver_torque, TORQUE_NUM_CPUS_PER_NODE)));
    queue_driver_free(driver_torque);
}

TEST_CASE("job_queue_set_driver", "[job_queue]") {
    job_queue_set_driver_(LSF_DRIVER);
    job_queue_set_driver_(LOCAL_DRIVER);
    job_queue_set_driver_(TORQUE_DRIVER);
    job_queue_set_driver_(SLURM_DRIVER);
}
