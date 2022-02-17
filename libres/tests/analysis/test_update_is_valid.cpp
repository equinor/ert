#include <filesystem>

#include <ert/analysis/update.hpp>
#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/state_map.hpp>
#include <ert/enkf/local_updatestep.hpp>
#include <ert/enkf/local_ministep.hpp>
#include <ert/analysis/analysis_module.hpp>

#include <catch2/catch.hpp>

namespace analysis {
bool is_valid(const analysis_config_type *analysis_config,
              const state_map_type *source_state_map, const int ensemble_size,
              const LocalUpdateStep *updatestep);
}

void analysis_config_set_min_realisations(analysis_config_type *config,
                                          int min_realisations);

void set_state_map(state_map_type *state_map, const int index,
                   realisation_state_enum state) {
    state_map_iset(state_map, index, STATE_INITIALIZED);
    state_map_iset(state_map, index, state);
}

void set_option_flag(analysis_module_type *analysis_module,
                     const analysis_module_flag_enum flag) {
    analysis_module_get_module_config(analysis_module)->set_option(flag);
}

TEST_CASE("is_valid", "[analysis]") {

    GIVEN("Allocated memory and default value parametrs") {
        // alloc memory for parameters
        analysis_config_type *analysis_config = analysis_config_alloc_default();
        state_map_type *source_state_map = state_map_alloc();
        LocalUpdateStep updatestep("name_not_important");
        const int ensemble_size = 1;
        analysis_module_type *analysis_module =
            analysis_module_alloc(ensemble_size, ENSEMBLE_SMOOTHER);
        LocalMinistep ministep("not-important");
        updatestep.add_ministep(&ministep);

        THEN("Assertion fail no active realization") {
            REQUIRE(analysis_config_get_min_realisations(analysis_config) == 0);
            REQUIRE(state_map_get_size(source_state_map) == 0);
            REQUIRE(updatestep.size() == 1);
            REQUIRE(!analysis::is_valid(analysis_config, source_state_map,
                                        ensemble_size, &updatestep));
        }

        WHEN("State is set to STATE_HAVE_DATA") {
            set_state_map(source_state_map, 0, STATE_HAS_DATA);
            THEN("Assertion pass") {
                REQUIRE(analysis::is_valid(analysis_config, source_state_map,
                                           ensemble_size, &updatestep));
            }
        }

        WHEN("Any state but STATE_HAS_DATA and STATE_UNDEFINED is set") {
            realisation_state_enum states[3] = {
                STATE_INITIALIZED, STATE_LOAD_FAILURE, STATE_PARENT_FAILURE};
            for (auto state : states) {
                set_state_map(source_state_map, 0, state);
                THEN("Assertion fail no active realization") {
                    REQUIRE(!analysis::is_valid(analysis_config,
                                                source_state_map, ensemble_size,
                                                &updatestep));
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
                REQUIRE(analysis::is_valid(analysis_config, source_state_map,
                                           ensemble_size, &updatestep));
            }

            THEN("Assertion fail with realizations < ensemble_size") {
                REQUIRE(!analysis::is_valid(analysis_config, source_state_map,
                                            2, &updatestep));
            }
        }

        WHEN("0 > realizations > min_realizations") {
            analysis_config_set_min_realisations(analysis_config, 2);
            set_state_map(source_state_map, 0, STATE_HAS_DATA);
            THEN("Assertion pass, realization >= ensemble_size") {
                REQUIRE(analysis::is_valid(analysis_config, source_state_map,
                                           ensemble_size, &updatestep));
            }
            THEN("Assertion fail, realization < ensemble_size") {
                REQUIRE(!analysis::is_valid(analysis_config, source_state_map,
                                            2, &updatestep));
            }
        }

        WHEN("ENSEMBLE_SMOOTER mode and Ministep > 1") {
            set_state_map(source_state_map, 0, STATE_HAS_DATA);
            analysis_module_type *other_analysis_module =
                analysis_module_alloc(ensemble_size, ENSEMBLE_SMOOTHER);
            LocalMinistep other_ministep("not-important");
            updatestep.add_ministep(&other_ministep);
            THEN("Assertion passes with no flags set") {
                REQUIRE(analysis::is_valid(analysis_config, source_state_map,
                                           ensemble_size, &updatestep));
            }
            for (const auto flag : {ANALYSIS_USE_A, ANALYSIS_UPDATE_A}) {
                set_option_flag(other_analysis_module, flag);
            }
            THEN("assert passes for none ANALYSIS_ITERABLE flags") {
                REQUIRE(analysis::is_valid(analysis_config, source_state_map,
                                           ensemble_size, &updatestep));
            }
            analysis_module_free(other_analysis_module);
        }

        WHEN("ITERATED_ENSEMBLE_SMOOTHER mode") {
            set_state_map(source_state_map, 0, STATE_HAS_DATA);
            LocalUpdateStep iter_updatestep("iter_update_step");
            analysis_module_type *iter_analysis_module = analysis_module_alloc(
                ensemble_size, ITERATED_ENSEMBLE_SMOOTHER);
            LocalMinistep iter_ministep("iter");
            iter_updatestep.add_ministep(&iter_ministep);
            THEN("Assertion passes when ministep count == 1") {
                REQUIRE(analysis::is_valid(analysis_config, source_state_map,
                                           ensemble_size, &iter_updatestep));
            }
            analysis_module_free(iter_analysis_module);
        }

        // cleanup
        analysis_module_free(analysis_module);
        state_map_free(source_state_map);
        analysis_config_free(analysis_config);
    }
}
