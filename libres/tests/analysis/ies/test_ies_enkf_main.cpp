#include <catch2/catch.hpp>

#include <ert/util/rng.h>
#include <ert/util/util.h>

#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_data.hpp>

TEST_CASE("ies_enkf_linalg_extract_active_E", "[analysis]") {
    int obs_size = 3;
    int ens_size = 2;

    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    ies::data::Data data(ens_size, true);

    // Initialising masks such that all observations and realizations are active
    bool_vector_type *ens_mask = bool_vector_alloc(ens_size, true);
    data.update_ens_mask(ens_mask);

    bool_vector_type *obs_mask = bool_vector_alloc(obs_size, true);
    data.store_initial_obs_mask(obs_mask);
    data.update_obs_mask(obs_mask);

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
    data.store_initialE(Ein);

    SECTION("ies_enkf_linalg_extract_active() does nothing when all "
            "observations and realizations are active") {
        auto *E = data.alloc_activeE();
        REQUIRE(matrix_equal(Ein, E));
        matrix_free(E);
    }

    SECTION("deactivate one realisation") {
        bool_vector_iset(ens_mask, 1, false);
        data.update_ens_mask(ens_mask);

        auto *E = data.alloc_activeE();
        REQUIRE(matrix_get_rows(E) == 3);
        REQUIRE(matrix_get_columns(E) == 1);

        REQUIRE(matrix_iget(E, 0, 0) == 1.0);
        REQUIRE(matrix_iget(E, 1, 0) == 2.0);
        REQUIRE(matrix_iget(E, 2, 0) == 3.0);
        matrix_free(E);
    }

    SECTION("deactivate one observation") {
        bool_vector_iset(obs_mask, 1, false);
        data.update_obs_mask(obs_mask);

        auto *E = data.alloc_activeE();
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
        data.update_obs_mask(obs_mask);

        bool_vector_iset(ens_mask, 1, false);
        data.update_ens_mask(ens_mask);

        auto *E = data.alloc_activeE();
        REQUIRE(matrix_get_rows(E) == 2);
        REQUIRE(matrix_get_columns(E) == 1);

        REQUIRE(matrix_iget(E, 0, 0) == 1.0);
        REQUIRE(matrix_iget(E, 1, 0) == 3.0);
        matrix_free(E);
    }

    matrix_free(Ein);
    bool_vector_free(ens_mask);
    bool_vector_free(obs_mask);
    rng_free(rng);
}

TEST_CASE("ies_enkf_linalg_extract_active_W", "[analysis]") {
    const int ens_size = 4;
    const int obs_size = 10;
    ies::data::Data data(ens_size, true);
    bool_vector_type *ens_mask = bool_vector_alloc(ens_size, true);
    bool_vector_type *obs_mask = bool_vector_alloc(obs_size, true);

    ies::init_update(&data, ens_mask, obs_mask, nullptr, nullptr, nullptr,
                     nullptr);
    data.update_ens_mask(ens_mask);

    matrix_type *W0 = matrix_alloc(ens_size, ens_size);
    for (int i = 0; i < ens_size; i++) {
        for (int j = 0; j < ens_size; j++)
            matrix_iset(W0, i, j, i * ens_size + j);
    }
    ies::linalg_store_active_W(&data, W0);

    {
        auto *W = data.alloc_activeW();
        REQUIRE(matrix_equal(W, W0));
        matrix_free(W);
    }

    // Deactivate one realization
    bool_vector_iset(ens_mask, 1, false);
    data.update_ens_mask(ens_mask);
    {
        auto *W = data.alloc_activeW();
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
}

SCENARIO("ies_enkf_linalg_extract_active_A", "[analysis]") {
    GIVEN("Inital setup") {
        const int ens_size = 4;
        const int obs_size = 10;
        const int state_size = 10;
        ies::data::Data data(ens_size, true);
        bool_vector_type *ens_mask = bool_vector_alloc(ens_size, true);
        bool_vector_type *obs_mask = bool_vector_alloc(obs_size, true);
        matrix_type *A0 = matrix_alloc(state_size, ens_size);
        for (int i = 0; i < state_size; i++) {
            for (int j = 0; j < ens_size; j++)
                matrix_iset(A0, i, j, i * ens_size + j);
        }
        ies::init_update(&data, ens_mask, obs_mask, nullptr, nullptr, nullptr,
                         nullptr);
        data.store_initialA(A0);

        WHEN("All realizations active") {
            auto *A = data.alloc_activeA();
            REQUIRE(matrix_equal(A, A0));
            matrix_free(A);
        }

        WHEN("One realization deactivated") {
            int dead_iens = 2;
            bool_vector_iset(ens_mask, dead_iens, false);
            data.update_ens_mask(ens_mask);
            auto *A = data.alloc_activeA();
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
    }
}
