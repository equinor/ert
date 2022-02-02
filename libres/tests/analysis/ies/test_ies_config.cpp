#include <catch2/catch.hpp>

#include <ert/analysis/ies/ies_config.hpp>
#include <ert/analysis/analysis_module.hpp>

TEST_CASE("ies_config", "[analysis]") {
    auto *config = ies::config::alloc(true);
    ies::config::set_option_flags(config, 0);

    SECTION("set_and_get_option") {
        REQUIRE(ies::config::get_option(config, ANALYSIS_NEED_ED) == false);
        ies::config::set_option(config, ANALYSIS_NEED_ED);
        REQUIRE(ies::config::get_option(config, ANALYSIS_NEED_ED) == true);

        ies::config::set_option(config, ANALYSIS_NEED_ED);
        REQUIRE(ies::config::get_option_flags(config) == ANALYSIS_NEED_ED);
    }

    SECTION("del_option") {
        ies::config::set_option(config, ANALYSIS_NEED_ED);
        REQUIRE(ies::config::get_option_flags(config) == ANALYSIS_NEED_ED);

        ies::config::del_option(config, ANALYSIS_NEED_ED);
        REQUIRE(ies::config::get_option(config, ANALYSIS_NEED_ED) == false);
        REQUIRE(ies::config::get_option_flags(config) == 0);

        ies::config::del_option(config, ANALYSIS_NEED_ED);
        REQUIRE(ies::config::get_option_flags(config) == 0);
    }

    ies::config::free(config);
}
