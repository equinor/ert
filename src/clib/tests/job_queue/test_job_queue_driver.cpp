#include "catch2/catch.hpp"
#include <cstdlib>
#include <string>

#include <ert/job_queue/job_queue.hpp>
#include <ert/job_queue/local_driver.hpp>
#include <ert/job_queue/lsf_driver.hpp>
#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/slurm_driver.hpp>

void job_queue_set_driver_(job_driver_type driver_type) {
    queue_driver_type *driver = queue_driver_alloc(driver_type);
    job_queue_type *queue = job_queue_alloc(driver);
    job_queue_free(queue);
    queue_driver_free(driver);
}

TEST_CASE("job_queue_set_driver", "[job_queue]") {
    job_queue_set_driver_(LSF_DRIVER);
    job_queue_set_driver_(LOCAL_DRIVER);
    job_queue_set_driver_(SLURM_DRIVER);
}
