#include <catch2/catch.hpp>

#include <ert/analysis/std_enkf.hpp>
#include <ert/analysis/ies/ies_data.hpp>
#include <ert/analysis/ies/ies_config.hpp>
#include <ert/analysis/analysis_module.hpp>

SCENARIO("Can set inversion method", "[std_enkf]") {
    GIVEN("A default constructed std_enkf_data instance") {
        auto *analysis_module = analysis_module_alloc(ENSEMBLE_SMOOTHER);

        WHEN("Setting invalid key") {
            REQUIRE(!analysis_module_set_var(analysis_module, "NO_SUCH_KEY",
                                             "VALUE"));
        }

        WHEN("Setting invalid value") {
            REQUIRE(!analysis_module_set_var(
                analysis_module, ies::config::INVERSION_KEY, "INVALID_VALUE"));
        }

        WHEN("Inversion is set to SUBSPACE_EXACT_R") {

            REQUIRE(analysis_module_set_var(
                analysis_module, ies::config::INVERSION_KEY,
                ies::config::STRING_INVERSION_SUBSPACE_EXACT_R));
            //REQUIRE(analysis_module_get_inversion(analysis_module) ==
            //        ies::config::IES_INVERSION_SUBSPACE_EXACT_R);
        }

        WHEN("Inversion is set to SUBSPACE_EE_R") {

            REQUIRE(analysis_module_set_var(
                analysis_module, ies::config::INVERSION_KEY,
                ies::config::STRING_INVERSION_SUBSPACE_EE_R));
            //REQUIRE(analysis_module_get_inversion(analysis_module) ==
            //        ies::config::IES_INVERSION_SUBSPACE_EE_R);
        }

        WHEN("Inversion is set to SUBSPACE_RE") {

            REQUIRE(analysis_module_set_var(
                analysis_module, ies::config::INVERSION_KEY,
                ies::config::STRING_INVERSION_SUBSPACE_RE));
            //REQUIRE(analysis_module_get_inversion(analysis_module) ==
            //        ies::config::IES_INVERSION_SUBSPACE_RE);
        }

        analysis_module_free(analysis_module);
    }
}
