#include "catch2/catch.hpp"
#include <ert/job_queue/slurm_driver.hpp>
#include <stdlib.h>

void test_option(slurm_driver_type *driver, const char *option,
                 const char *value) {
    REQUIRE(slurm_driver_set_option(driver, option, value));
    REQUIRE(std::string((const char *)slurm_driver_get_option(
                driver, option)) == std::string(value));
}

void test_host_options(slurm_driver_type *driver, const char *option) {
    REQUIRE(slurm_driver_set_option(driver, option, "host1"));
    REQUIRE(std::string((const char *)slurm_driver_get_option(
                driver, option)) == "host1");

    REQUIRE(slurm_driver_set_option(driver, option, "host2"));
    REQUIRE(std::string((const char *)slurm_driver_get_option(
                driver, option)) == "host1,host2");

    REQUIRE(slurm_driver_set_option(driver, option, "host2 host3,host4"));
    REQUIRE(std::string((const char *)slurm_driver_get_option(
                driver, option)) == "host1,host2,host3,host4");
}

TEST_CASE("job_slurm_options", "[job_slurm]") {
    slurm_driver_type *driver = (slurm_driver_type *)slurm_driver_alloc();
    test_option(driver, SLURM_PARTITION_OPTION, "my_partition");
    test_option(driver, SLURM_SBATCH_OPTION, "my_funny_sbatch");
    test_option(driver, SLURM_SCANCEL_OPTION, "my_funny_scancel");
    test_option(driver, SLURM_SQUEUE_OPTION, "my_funny_squeue");
    test_option(driver, SLURM_SCONTROL_OPTION, "my_funny_scontrol");
    test_option(driver, SLURM_SQUEUE_TIMEOUT_OPTION, "11");
    test_option(driver, SLURM_MAX_RUNTIME_OPTION, "11");
    test_option(driver, SLURM_MEMORY_OPTION, "100mb");
    test_option(driver, SLURM_MEMORY_PER_CPU_OPTION, "1000gb");
    REQUIRE_FALSE(slurm_driver_set_option(driver, "SLURM_SQUEUE_TIMEOUT_OPTION",
                                          "NOT_INTEGER"));
    REQUIRE_FALSE(slurm_driver_set_option(driver, "NO_SUCH_OPTION", "Value"));
    test_host_options(driver, SLURM_INCLUDE_HOST_OPTION);
    test_host_options(driver, SLURM_EXCLUDE_HOST_OPTION);
    slurm_driver_free(driver);
}
