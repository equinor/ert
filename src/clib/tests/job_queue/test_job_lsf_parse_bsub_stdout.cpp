#include "catch2/catch.hpp"
#include <cstdlib>
#include <ert/job_queue/lsf_driver.hpp>

TEST_CASE("job_lsf_parse_bsub_empty_file", "[job_lsf_parse_bsub_stdout]") {
    REQUIRE(lsf_job_parse_bsub_stdout("") == -1);
}

TEST_CASE("job_lsf_parse_bsub_status_ok", "[job_lsf_parse_bsub_stdout]") {
    REQUIRE(lsf_job_parse_bsub_stdout(
                "Job <12345> is submitted to default queue <normal>.\n") ==
            12345);
}
