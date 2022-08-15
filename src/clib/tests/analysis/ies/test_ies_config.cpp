#include <catch2/catch.hpp>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/ies/ies_config.hpp>

TEST_CASE("ies_config", "[analysis]") {
    ies::Config config(true);
    config.set_option_flags(0);

    SECTION("set_and_get_option") {
        REQUIRE(config.get_option(ANALYSIS_UPDATE_A) == false);
        config.set_option(ANALYSIS_UPDATE_A);
        REQUIRE(config.get_option(ANALYSIS_UPDATE_A) == true);

        config.set_option(ANALYSIS_UPDATE_A);
        REQUIRE(config.get_option_flags() == ANALYSIS_UPDATE_A);
    }

    SECTION("del_option") {
        config.set_option(ANALYSIS_UPDATE_A);
        REQUIRE(config.get_option_flags() == ANALYSIS_UPDATE_A);

        config.del_option(ANALYSIS_UPDATE_A);
        REQUIRE(config.get_option(ANALYSIS_UPDATE_A) == false);
        REQUIRE(config.get_option_flags() == 0);

        config.del_option(ANALYSIS_UPDATE_A);
        REQUIRE(config.get_option_flags() == 0);
    }
}
