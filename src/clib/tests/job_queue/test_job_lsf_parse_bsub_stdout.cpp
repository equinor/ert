#include "catch2/catch.hpp"
#include <cstdlib>
#include <ert/job_queue/lsf_driver.hpp>

TEST_CASE("job_lsf_parse_bsub_empty_file", "[job_lsf_parse_bsub_stdout]") {
    const char *stdout_file = "bsub_empty";
    FILE *stream = fopen(stdout_file, "w");
    REQUIRE(stream != nullptr);
    fclose(stream);
    REQUIRE(lsf_job_parse_bsub_stdout("bsub", stdout_file) == -1);
}

TEST_CASE("job_lsf_parse_bsub_status_ok", "[job_lsf_parse_bsub_stdout]") {
    const char *stdout_file = "bsub_OK";
    FILE *stream = fopen(stdout_file, "w");
    REQUIRE(stream != nullptr);
    fprintf(stream, "Job <12345> is submitted to default queue <normal>.\n");
    fclose(stream);
    REQUIRE(lsf_job_parse_bsub_stdout("bsub", stdout_file) == 12345);
}

TEST_CASE("job_lsf_parse_bsub_no_file", "[job_lsf_parse_bsub_stdout]") {
    REQUIRE(lsf_job_parse_bsub_stdout("bsub", "does/not/exist") == -1);
}
