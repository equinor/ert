#include <catch2/catch.hpp>

#include <ert/util/rng.h>
#include <ert/util/util.h>

#include <ies_enkf.hpp>
#include <ies_enkf_data.hpp>

void ies_enkf_linalg_extract_active(const ies_enkf_data_type *data,
                                    matrix_type *E, FILE *log_fp, bool dbg);

TEST_CASE("ies_enkf_linalg_extract_active", "[analysis]") {
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    ies_enkf_data_type *data = (ies_enkf_data_type *)ies_enkf_data_alloc();

    int state_size = 3;
    int ens_size = 2;

    // Initialising masks such that all observations and realizations are active
    bool_vector_type *ens_mask = bool_vector_alloc(ens_size, true);
    ies_enkf_data_update_ens_mask(data, ens_mask);

    bool_vector_type *obs_mask = bool_vector_alloc(state_size, true);
    ies_enkf_update_obs_mask(data, obs_mask);
    ies_enkf_store_initial_obs_mask(data, obs_mask);

    // Set initial data
    matrix_type *Ein = matrix_alloc(state_size, ens_size);

    // Set first column
    matrix_iset(Ein, 0, 0, 1.0);
    matrix_iset(Ein, 1, 0, 2.0);
    matrix_iset(Ein, 2, 0, 3.0);

    // Set second column
    matrix_iset(Ein, 0, 1, 1.5);
    matrix_iset(Ein, 1, 1, 2.5);
    matrix_iset(Ein, 2, 1, 3.5);

    // minimal config needed to set initial data of `iens_enkf_data_type`
    ies_enkf_config_set_ies_debug(ies_enkf_data_get_config(data), false);
    ies_enkf_config_set_ies_logfile(ies_enkf_data_get_config(data),
                                    "log_test_ies_enkf_linalg_extract_active");
    FILE *test_log = ies_enkf_data_open_log(data);
    ies_enkf_data_store_initialE(data, Ein);
    ies_enkf_data_fclose_log(data);

    // Test that `ies_enkf_linalg_extract_active` does nothing when all observations and realizations are active
    matrix_type *E = matrix_alloc(state_size, ens_size);
    ies_enkf_linalg_extract_active(data, E, stdout, false);
    REQUIRE(matrix_equal(Ein, E));

    // Test that `ies_enkf_linalg_extract_active` can deactivate an ensemble
    bool_vector_iset(ens_mask, 1, false);
    ies_enkf_data_update_ens_mask(data, ens_mask);

    matrix_type *E_ens_deactivate = matrix_alloc(state_size, ens_size);
    ies_enkf_linalg_extract_active(data, E_ens_deactivate, stdout, false);

    REQUIRE(matrix_iget(E_ens_deactivate, 0, 0) == 1.0);
    REQUIRE(matrix_iget(E_ens_deactivate, 1, 0) == 2.0);
    REQUIRE(matrix_iget(E_ens_deactivate, 2, 0) == 3.0);
    REQUIRE(matrix_iget(E_ens_deactivate, 0, 1) == 0.0);
    REQUIRE(matrix_iget(E_ens_deactivate, 1, 1) == 0.0);
    REQUIRE(matrix_iget(E_ens_deactivate, 2, 1) == 0.0);

    // Test that `ies_enkf_linalg_extract_active` can deactivate an observation
    bool_vector_iset(obs_mask, 1, false);
    ies_enkf_update_obs_mask(data, obs_mask);

    matrix_type *E_obs_deactivate = matrix_alloc(state_size, ens_size);
    ies_enkf_linalg_extract_active(data, E_obs_deactivate, stdout, false);

    REQUIRE(matrix_iget(E_obs_deactivate, 0, 0) == 1.0);
    REQUIRE(matrix_iget(E_obs_deactivate, 1, 0) == 3.0);
    REQUIRE(matrix_iget(E_obs_deactivate, 2, 0) == 0.0);
    REQUIRE(matrix_iget(E_obs_deactivate, 0, 1) == 0.0);
    REQUIRE(matrix_iget(E_obs_deactivate, 1, 1) == 0.0);
    REQUIRE(matrix_iget(E_obs_deactivate, 2, 1) == 0.0);

    bool_vector_free(ens_mask);
    bool_vector_free(obs_mask);
    matrix_free(E);
    matrix_free(E_ens_deactivate);
    matrix_free(E_obs_deactivate);
    ies_enkf_data_free(data);
    rng_free(rng);
}
