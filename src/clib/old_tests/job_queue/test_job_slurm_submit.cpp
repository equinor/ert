#include <stdlib.h>

#include <vector>

#include <ert/util/test_util.hpp>
#include <ert/util/test_work_area.hpp>
#include <ert/util/util.hpp>

#include <cassert>
#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/slurm_driver.hpp>

void make_sleep_job(const char *fname, int sleep_time) {
    FILE *stream = fopen(fname, "w");
    assert(stream);
    fprintf(stream, "sleep %d \n", sleep_time);
    fclose(stream);

    mode_t fmode = S_IRWXU;
    chmod(fname, fmode);
}

void submit_job(queue_driver_type *driver, const ecl::util::TestArea &ta,
                const std::string &job_name, const char *cmd,
                bool expect_fail) {
    std::string run_path = ta.test_cwd() + "/" + job_name;
    util_make_path(run_path.c_str());
    auto job = queue_driver_submit_job(driver, cmd, 1, run_path.c_str(),
                                       job_name.c_str(), 0, nullptr);
    if (expect_fail)
        test_assert_NULL(job);
    else {
        test_assert_not_NULL(job);
        queue_driver_kill_job(driver, job);
        queue_driver_free_job(driver, job);
    }
}

void run() {
    ecl::util::TestArea ta("slurm_submit", true);
    queue_driver_type *driver = queue_driver_alloc(SLURM_DRIVER);
    const char *cmd = util_alloc_abs_path("cmd.sh");

    make_sleep_job(cmd, 10);
    submit_job(driver, ta, "JOB1", cmd, false);
    queue_driver_set_option(driver, SLURM_PARTITION_OPTION,
                            "invalid_partition");
    submit_job(driver, ta, "JOB1", cmd, true);

    queue_driver_free(driver);
}

int main(int argc, char **argv) { run(); }
