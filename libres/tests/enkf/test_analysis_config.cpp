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

TEST_CASE("Accessing analysis modules loaded in config", "[enkf]") {
    GIVEN("A default analysis config with internal modules loaded") {
        auto analysis_config = analysis_config_alloc_default();
        WHEN("Internal modules are loaded") {
            analysis_config_load_internal_modules(analysis_config);
            THEN("Fetching existing module do not raise exception") {
                REQUIRE_NOTHROW(
                    analysis_config_get_module(analysis_config, "STD_ENKF"));
            }
            THEN("Fetching non existing module raises exception") {
                REQUIRE_THROWS(
                    analysis_config_get_module(analysis_config, "UNKNOWN"));
            }
            analysis_config_set_single_node_update(analysis_config, true);
            THEN("Feching ies module returns false") {
                REQUIRE(analysis_config_select_module(analysis_config,
                                                      "IES_ENKF") == false);
            }
        }
    }
}
