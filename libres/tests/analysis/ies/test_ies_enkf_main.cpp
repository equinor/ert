#include <catch2/catch.hpp>

#include <ert/util/rng.h>
#include <ert/util/util.h>

#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_data.hpp>

TEST_CASE("ies_enkf_linalg_extract_active_E", "[analysis]") {
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    auto *data = static_cast<ies::data_type *>(ies::data_alloc());

    int obs_size = 3;
    int ens_size = 2;

    // Initialising masks such that all observations and realizations are active
    bool_vector_type *ens_mask = bool_vector_alloc(ens_size, true);
    ies::data_update_ens_mask(data, ens_mask);

    bool_vector_type *obs_mask = bool_vector_alloc(obs_size, true);
    ies::update_obs_mask(data, obs_mask);
    ies::store_initial_obs_mask(data, obs_mask);

    // Set initial data
    matrix_type *Ein = matrix_alloc(obs_size, ens_size);

    // Set first column
    matrix_iset(Ein, 0, 0, 1.0);
    matrix_iset(Ein, 1, 0, 2.0);
    matrix_iset(Ein, 2, 0, 3.0);

    // Set second column
    matrix_iset(Ein, 0, 1, 1.5);
    matrix_iset(Ein, 1, 1, 2.5);
    matrix_iset(Ein, 2, 1, 3.5);
    ies::data_store_initialE(data, Ein);

    SECTION("ies_enkf_linalg_extract_active() does nothing when all "
            "observations and realizations are active") {
        matrix_type *E = ies::alloc_activeE(data);
        REQUIRE(matrix_equal(Ein, E));
        matrix_free(E);
    }

    SECTION("deactivate one realisation") {
        bool_vector_iset(ens_mask, 1, false);
        ies::data_update_ens_mask(data, ens_mask);

        matrix_type *E = ies::alloc_activeE(data);
        REQUIRE(matrix_get_rows(E) == 3);
        REQUIRE(matrix_get_columns(E) == 1);

        REQUIRE(matrix_iget(E, 0, 0) == 1.0);
        REQUIRE(matrix_iget(E, 1, 0) == 2.0);
        REQUIRE(matrix_iget(E, 2, 0) == 3.0);
        matrix_free(E);
    }

    SECTION("deactivate one observation") {
        bool_vector_iset(obs_mask, 1, false);
        ies::update_obs_mask(data, obs_mask);

        matrix_type *E = ies::alloc_activeE(data);
        REQUIRE(matrix_get_rows(E) == 2);
        REQUIRE(matrix_get_columns(E) == 2);

        REQUIRE(matrix_iget(E, 0, 0) == 1.0);
        REQUIRE(matrix_iget(E, 1, 0) == 3.0);
        REQUIRE(matrix_iget(E, 0, 1) == 1.5);
        REQUIRE(matrix_iget(E, 1, 1) == 3.5);
        matrix_free(E);
    }

    SECTION("deactivate one observation and one realisation") {
        bool_vector_iset(obs_mask, 1, false);
        ies::update_obs_mask(data, obs_mask);

        bool_vector_iset(ens_mask, 1, false);
        ies::data_update_ens_mask(data, ens_mask);

        matrix_type *E = ies::alloc_activeE(data);
        REQUIRE(matrix_get_rows(E) == 2);
        REQUIRE(matrix_get_columns(E) == 1);

        REQUIRE(matrix_iget(E, 0, 0) == 1.0);
        REQUIRE(matrix_iget(E, 1, 0) == 3.0);
        matrix_free(E);
    }

    matrix_free(Ein);
    bool_vector_free(ens_mask);
    bool_vector_free(obs_mask);
    ies::data_free(data);
    rng_free(rng);
}

TEST_CASE("ies_enkf_linalg_extract_active_W", "[analysis]") {
    const int ens_size = 4;
    const int obs_size = 10;
    auto *data = static_cast<ies::data_type *>(ies::data_alloc());
    bool_vector_type *ens_mask = bool_vector_alloc(ens_size, true);
    bool_vector_type *obs_mask = bool_vector_alloc(obs_size, true);

    ies::config_set_logfile(ies::data_get_config(data), "log");
    ies::data_open_log(data);
    ies::init_update(data, ens_mask, obs_mask, nullptr, nullptr, nullptr,
                     nullptr, nullptr, nullptr);
    ies::data_update_ens_mask(data, ens_mask);

    matrix_type *W0 = matrix_alloc(ens_size, ens_size);
    for (int i = 0; i < ens_size; i++) {
        for (int j = 0; j < ens_size; j++)
            matrix_iset(W0, i, j, i * ens_size + j);
    }
    ies::linalg_store_active_W(data, W0);

    {
        matrix_type *W = ies::alloc_activeW(data);
        REQUIRE(matrix_equal(W, W0));
        matrix_free(W);
    }

    // Deactivate one realization
    bool_vector_iset(ens_mask, 1, false);
    ies::data_update_ens_mask(data, ens_mask);
    {
        matrix_type *W = ies::alloc_activeW(data);
        for (int i = 0; i < ens_size - 1; i++) {
            for (int j = 0; j < ens_size - 1; j++) {
                int i0 = i + (i > 0);
                int j0 = j + (j > 0);
                REQUIRE(matrix_iget(W, i, j) == matrix_iget(W0, i0, j0));
            }
        }
        matrix_free(W);
    }

    matrix_free(W0);
    bool_vector_free(ens_mask);
    bool_vector_free(obs_mask);
    ies::data_free(data);
}

SCENARIO("ies_enkf_linalg_extract_active_A", "[analysis]") {
    GIVEN("Inital setup") {
        const int ens_size = 4;
        const int obs_size = 10;
        const int state_size = 10;
        auto *data = static_cast<ies::data_type *>(ies::data_alloc());
        bool_vector_type *ens_mask = bool_vector_alloc(ens_size, true);
        bool_vector_type *obs_mask = bool_vector_alloc(obs_size, true);
        matrix_type *A0 = matrix_alloc(state_size, ens_size);
        for (int i = 0; i < state_size; i++) {
            for (int j = 0; j < ens_size; j++)
                matrix_iset(A0, i, j, i * ens_size + j);
        }
        ies::init_update(data, ens_mask, obs_mask, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr);
        ies::data_store_initialA(data, A0);

        WHEN("All realizations active") {
            matrix_type *A = ies::alloc_activeA(data);
            REQUIRE(matrix_equal(A, A0));
            matrix_free(A);
        }

        WHEN("One realization deactivated") {
            int dead_iens = 2;
            bool_vector_iset(ens_mask, dead_iens, false);
            ies::data_update_ens_mask(data, ens_mask);
            matrix_type *A = ies::alloc_activeA(data);
            for (int i = 0; i < state_size; i++) {
                int i0 = i;
                for (int j = 0; j < ens_size - 1; j++) {
                    int j0 = j;
                    if (j0 >= dead_iens)
                        j0 += 1;

                    REQUIRE(matrix_iget(A, i, j) == matrix_iget(A0, i0, j0));
                }
            }
            matrix_free(A);
        }

        matrix_free(A0);
        bool_vector_free(ens_mask);
        bool_vector_free(obs_mask);
        ies::data_free(data);
    }
}
