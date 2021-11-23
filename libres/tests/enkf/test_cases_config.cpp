#include <filesystem>

#include "catch2/catch.hpp"

#include <ert/enkf/cases_config.hpp>

#include "../tmpdir.hpp"

namespace fs = std::filesystem;

TEST_CASE("cases_config", "[enkf]") {
    GIVEN("A cases config with iteration number") {
        TmpDir tmpdir;
        cases_config_type *cases_config = cases_config_alloc();
        int iteration_number = GENERATE(1, 2, 3, 14);
        cases_config_set_int(cases_config, "iteration_number",
                             iteration_number);

        WHEN("The cases config is written and then read back") {
            fs::path file_path = fs::current_path() / "TEST_CASES_CONFIG";
            cases_config_fwrite(cases_config, file_path.c_str());
            cases_config_fread(cases_config, file_path.c_str());

            THEN("Iteration number remains the same") {
                REQUIRE(cases_config_get_iteration_number(cases_config) ==
                        iteration_number);
            }
        }
        cases_config_free(cases_config);
    }
}
