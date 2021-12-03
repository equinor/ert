#include "catch2/catch.hpp"

#include <ert/enkf/analysis_config.hpp>
#include <ert/util/stringlist.h>

#include "../tmpdir.hpp"

void analysis_config_load_internal_modules(analysis_config_type *config);

TEST_CASE("analysis_config_module_names", "[enkf]") {
    GIVEN("A default analysis config with internal modules loaded") {
        auto analysis_config = analysis_config_alloc_default();
        analysis_config_load_internal_modules(analysis_config);

        WHEN("Module names are allocated") {
            auto modules = analysis_config_module_names(analysis_config);

            THEN("The list should contain names of internal modules") {
                REQUIRE(modules.size() == 2);
                REQUIRE(modules.at(0) == "STD_ENKF");
                REQUIRE(modules.at(1) == "IES_ENKF");
            }
        }
    }
}
