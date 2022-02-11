#include <catch2/catch.hpp>

#include <ert/analysis/ies/ies_config.hpp>
#include <ert/analysis/analysis_module.hpp>

TEST_CASE("ies_config", "[analysis]") {
    ies::config::Config config(true);
    config.set_option_flags(0);

    SECTION("set_and_get_option") {
        REQUIRE(config.get_option(ANALYSIS_NEED_ED) == false);
        config.set_option(ANALYSIS_NEED_ED);
        REQUIRE(config.get_option(ANALYSIS_NEED_ED) == true);

        config.set_option(ANALYSIS_NEED_ED);
        REQUIRE(config.get_option_flags() == ANALYSIS_NEED_ED);
    }

    SECTION("del_option") {
        config.set_option(ANALYSIS_NEED_ED);
        REQUIRE(config.get_option_flags() == ANALYSIS_NEED_ED);

        config.del_option(ANALYSIS_NEED_ED);
        REQUIRE(config.get_option(ANALYSIS_NEED_ED) == false);
        REQUIRE(config.get_option_flags() == 0);

        config.del_option(ANALYSIS_NEED_ED);
        REQUIRE(config.get_option_flags() == 0);
    }
}
