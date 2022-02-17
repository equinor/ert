#include <filesystem>

#include <ert/analysis/update.hpp>
#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/state_map.hpp>
#include <ert/enkf/local_updatestep.hpp>
#include <ert/enkf/local_ministep.hpp>
#include <ert/analysis/analysis_module.hpp>

#include <catch2/catch.hpp>

namespace analysis {
bool is_valid(const ies::config::Config &ies_config,
              const analysis_config_type *analysis_config,
              const state_map_type *source_state_map, const int ensemble_size,
              const local_updatestep_type *updatestep);
}

void analysis_config_set_min_realisations(analysis_config_type *config,
                                          int min_realisations);

void set_state_map(state_map_type *state_map, const int index,
                   realisation_state_enum state) {
    state_map_iset(state_map, index, STATE_INITIALIZED);
    state_map_iset(state_map, index, state);
}

void set_option_flag(ies::config::Config &ies_config,
                     const analysis_module_flag_enum flag) {
    ies_config.set_option(flag);
}

TEST_CASE("is_valid", "[analysis]") {

    GIVEN("Allocated memory and default value parametrs") {
        // alloc memory for parameters
        analysis_config_type *analysis_config = analysis_config_alloc_default();
        state_map_type *source_state_map = state_map_alloc();
        local_updatestep_type *updatestep =
            local_updatestep_alloc("name_not_important");
        const int ensemble_size = 1;
        analysis_module_type *analysis_module =
            analysis_module_alloc(ensemble_size, ENSEMBLE_SMOOTHER);
        auto *ies_config = analysis_module_get_module_config(analysis_module);
        local_ministep_type *ministep = local_ministep_alloc("not important");
        local_updatestep_add_ministep(updatestep, ministep);

        THEN("Assertion fail no active realization") {
            REQUIRE(analysis_config_get_min_realisations(analysis_config) == 0);
            REQUIRE(state_map_get_size(source_state_map) == 0);
            REQUIRE(local_updatestep_get_num_ministep(updatestep) == 1);
            REQUIRE(!analysis::is_valid(*ies_config, analysis_config,
                                        source_state_map, ensemble_size,
                                        updatestep));
        }

        WHEN("State is set to STATE_HAVE_DATA") {
            set_state_map(source_state_map, 0, STATE_HAS_DATA);
            THEN("Assertion pass") {
                REQUIRE(analysis::is_valid(*ies_config, analysis_config,
                                           source_state_map, ensemble_size,
                                           updatestep));
            }
        }

        WHEN("Any state but STATE_HAS_DATA and STATE_UNDEFINED is set") {
            realisation_state_enum states[3] = {
                STATE_INITIALIZED, STATE_LOAD_FAILURE, STATE_PARENT_FAILURE};
            for (auto state : states) {
                set_state_map(source_state_map, 0, state);
                THEN("Assertion fail no active realization") {
                    REQUIRE(!analysis::is_valid(*ies_config, analysis_config,
                                                source_state_map, ensemble_size,
                                                updatestep));
                }
            }
        }

        WHEN("one STATE_LOAD_FAILURE, but with others on STATE_HAS_DATA") {
            set_state_map(source_state_map, 0, STATE_LOAD_FAILURE);
            set_state_map(source_state_map, 1, STATE_HAS_DATA);
            REQUIRE(state_map_get_size(source_state_map) == 2);
            REQUIRE(state_map_count_matching(source_state_map,
                                             STATE_LOAD_FAILURE) == 1);

            THEN("Assertion pass with realizations >= ensemble_size") {
                REQUIRE(analysis::is_valid(*ies_config, analysis_config,
                                           source_state_map, ensemble_size,
                                           updatestep));
            }

            THEN("Assertion fail with realizations < ensemble_size") {
                REQUIRE(!analysis::is_valid(*ies_config, analysis_config,
                                            source_state_map, 2, updatestep));
            }
        }

        WHEN("0 > realizations > min_realizations") {
            analysis_config_set_min_realisations(analysis_config, 2);
            set_state_map(source_state_map, 0, STATE_HAS_DATA);
            THEN("Assertion pass, realization >= ensemble_size") {
                REQUIRE(analysis::is_valid(*ies_config, analysis_config,
                                           source_state_map, ensemble_size,
                                           updatestep));
            }
            THEN("Assertion fail, realization < ensemble_size") {
                REQUIRE(!analysis::is_valid(*ies_config, analysis_config,
                                            source_state_map, 2, updatestep));
            }
        }

        WHEN("ENSEMBLE_SMOOTER mode and Ministep > 1") {
            set_state_map(source_state_map, 0, STATE_HAS_DATA);
            analysis_module_type *other_analysis_module =
                analysis_module_alloc(ensemble_size, ENSEMBLE_SMOOTHER);
            local_ministep_type *other_ministep =
                local_ministep_alloc("other analysis module");
            auto *other_ies_config =
                analysis_module_get_module_config(other_analysis_module);
            local_updatestep_add_ministep(updatestep, other_ministep);
            THEN("Assertion passes with no flags set") {
                REQUIRE(analysis::is_valid(*other_ies_config, analysis_config,
                                           source_state_map, ensemble_size,
                                           updatestep));
            }
            for (const auto flag : {ANALYSIS_USE_A, ANALYSIS_UPDATE_A}) {
                set_option_flag(*ies_config, flag);
            }
            THEN("assert passes for none ANALYSIS_ITERABLE flags") {
                REQUIRE(analysis::is_valid(*other_ies_config, analysis_config,
                                           source_state_map, ensemble_size,
                                           updatestep));
            }
            local_ministep_free(other_ministep);
            analysis_module_free(other_analysis_module);
        }

        WHEN("ITERATED_ENSEMBLE_SMOOTHER mode") {
            set_state_map(source_state_map, 0, STATE_HAS_DATA);
            local_updatestep_type *iter_updatestep =
                local_updatestep_alloc("iter_update_step");
            analysis_module_type *iter_analysis_module = analysis_module_alloc(
                ensemble_size, ITERATED_ENSEMBLE_SMOOTHER);
            local_ministep_type *iter_ministep = local_ministep_alloc("iter");
            auto *iter_ies_config =
                analysis_module_get_module_config(iter_analysis_module);
            local_updatestep_add_ministep(updatestep, iter_ministep);
            THEN("Assertion passes when ministep count == 1") {
                REQUIRE(analysis::is_valid(*iter_ies_config, analysis_config,
                                           source_state_map, ensemble_size,
                                           iter_updatestep));
            }
            local_ministep_free(iter_ministep);
            analysis_module_free(iter_analysis_module);
            local_updatestep_free(iter_updatestep);
        }

        // cleanup
        local_ministep_free(ministep);
        analysis_module_free(analysis_module);
        local_updatestep_free(updatestep);
        state_map_free(source_state_map);
        analysis_config_free(analysis_config);
    }
}
