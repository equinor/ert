#include <stdlib.h>

#include <ert/util/test_util.hpp>

#include <ert/job_queue/slurm_driver.hpp>

void test_option(slurm_driver_type *driver, const char *option,
                 const char *value) {
    test_assert_true(slurm_driver_set_option(driver, option, value));
    test_assert_string_equal(
        (const char *)slurm_driver_get_option(driver, option), value);
}

void test_host_options(slurm_driver_type *driver, const char *option) {
    test_assert_true(slurm_driver_set_option(driver, option, "host1"));
    test_assert_string_equal(
        (const char *)slurm_driver_get_option(driver, option), "host1");

    test_assert_true(slurm_driver_set_option(driver, option, "host2"));
    test_assert_string_equal(
        (const char *)slurm_driver_get_option(driver, option), "host1,host2");

    test_assert_true(
        slurm_driver_set_option(driver, option, "host2 host3,host4"));
    test_assert_string_equal(
        (const char *)slurm_driver_get_option(driver, option),
        "host1,host2,host3,host4");
}

void test_options() {
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
    test_assert_false(slurm_driver_set_option(
        driver, "SLURM_SQUEUE_TIMEOUT_OPTION", "NOT_INTEGER"));
    test_assert_false(
        slurm_driver_set_option(driver, "NO_SUCH_OPTION", "Value"));
    test_host_options(driver, SLURM_INCLUDE_HOST_OPTION);
    test_host_options(driver, SLURM_EXCLUDE_HOST_OPTION);
    slurm_driver_free(driver);
}

int main(int argc, char **argv) {
    test_options();
    exit(0);
}
