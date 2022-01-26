#include "../tmpdir.hpp"
#include <catch2/catch.hpp>
#include <filesystem>
#include <fstream>
#include <ert/job_queue/ext_job.hpp>
#include <string>   // std::string
#include <iostream> // std::cout
#include <sstream>  // std::stringstream

SCENARIO("Loading ext jobs with different permissions") {
    GIVEN("A non existing executable") {
        WITH_TMPDIR;
        WHEN("Creating a job") {
            ext_job_type *job = ext_job_alloc("JOB", nullptr, false);
            THEN("setting the executable to a non existing file causes "
                 "exception") {
                REQUIRE_THROWS_WITH(
                    ext_job_set_executable(job, "non-existing", "non-existing",
                                           true),
                    Catch::Matchers::Contains("non-existing was not found"));
            }
        }
    }
    GIVEN("A file with non-executable rights") {
        WITH_TMPDIR;
        std::ofstream jobfile("executable");
        jobfile << "executable" << std::endl;
        jobfile.close();
        WHEN("Loading the file") {
            ext_job_type *job = ext_job_alloc("JOB", nullptr, false);
            THEN("setting the executable to a non-executable file causes "
                 "exception") {
                REQUIRE_THROWS_WITH(ext_job_set_executable(job, "executable",
                                                           "executable", true),
                                    Catch::Matchers::Contains(
                                        "You do not have execute rights"));
            }
        }
    }
    GIVEN("A file with executable rights") {
        WITH_TMPDIR;
        std::ofstream jobfile("executable");
        jobfile << "executable" << std::endl;
        jobfile.close();
        chmod("executable", 0777);
        WHEN("Loading the file") {
            ext_job_type *job = ext_job_alloc("JOB", nullptr, false);
            THEN("setting the executable") {
                REQUIRE_NOTHROW(ext_job_set_executable(job, "executable",
                                                       "executable", true));
                auto exec_path = std::filesystem::current_path() /
                                 std::filesystem::path("executable");
                auto job_exec_path =
                    std::filesystem::path(ext_job_get_executable(job));
                REQUIRE(exec_path == job_exec_path);
            }
        }
    }
}