#include <catch2/catch.hpp>

#include <ert/analysis/std_enkf.hpp>

SCENARIO("Can set inversion method", "[std_enkf]") {
    GIVEN("A default constructed std_enkf_data instance") {
        std_enkf_data_type *std_enkf_data =
            static_cast<std_enkf_data_type *>(std_enkf_data_alloc());

        WHEN("Setting invalid key") {
            REQUIRE(
                !std_enkf_set_string(std_enkf_data, "NO_SUCH_KEY", "VALUE"));
        }

        WHEN("Setting invalid value") {
            REQUIRE(!std_enkf_set_string(std_enkf_data, INVERSION_KEY,
                                         "INVALID_VALUE"));
        }

        WHEN("Inversion is set to SUBSPACE_EXACT_R") {

            REQUIRE(std_enkf_set_string(std_enkf_data, INVERSION_KEY,
                                        STRING_INVERSION_SUBSPACE_EXACT_R));
            REQUIRE(std_enkf_data_get_inversion(std_enkf_data) ==
                    ies::IES_INVERSION_SUBSPACE_EXACT_R);
        }

        WHEN("Inversion is set to SUBSPACE_EE_R") {

            REQUIRE(std_enkf_set_string(std_enkf_data, INVERSION_KEY,
                                        STRING_INVERSION_SUBSPACE_EE_R));
            REQUIRE(std_enkf_data_get_inversion(std_enkf_data) ==
                    ies::IES_INVERSION_SUBSPACE_EE_R);
        }

        WHEN("Inversion is set to SUBSPACE_RE") {

            REQUIRE(std_enkf_set_string(std_enkf_data, INVERSION_KEY,
                                        STRING_INVERSION_SUBSPACE_RE));
            REQUIRE(std_enkf_data_get_inversion(std_enkf_data) ==
                    ies::IES_INVERSION_SUBSPACE_RE);
        }

        WHEN("Deprecated bool flags are set to true") {
            std_enkf_set_bool(std_enkf_data, USE_EE_KEY_, false);
            std_enkf_set_bool(std_enkf_data, USE_GE_KEY_, false);
            REQUIRE(std_enkf_data_get_inversion(std_enkf_data) ==
                    ies::IES_INVERSION_SUBSPACE_EXACT_R);
        }

        WHEN("Deprecated bool flags aere set to false") {
            std_enkf_set_bool(std_enkf_data, USE_EE_KEY_, true);
            std_enkf_set_bool(std_enkf_data, USE_GE_KEY_, true);
            REQUIRE(std_enkf_data_get_inversion(std_enkf_data) ==
                    ies::IES_INVERSION_SUBSPACE_RE);
        }

        std_enkf_data_free(std_enkf_data);
    }
}
