#include <catch2/catch.hpp>
#include <vector>
#include <cmath>

#include <ert/util/rng.h>
#include <ert/enkf/enkf_util.hpp>

#include <ert/util/bool_vector.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/meas_data.hpp>
#include <ert/analysis/ies/ies_enkf.hpp>
#include <ert/analysis/std_enkf.hpp>

struct model {
    double amplitude;
    double phase;

    model(double a, double p) : amplitude(a), phase(p) {}

    model(rng_type *rng) {
        this->amplitude = 0.25 + enkf_util_rand_normal(1, 0.5, rng);
        this->phase = enkf_util_rand_normal(0, M_PI / 4, rng);
    }

    double eval(double x) const {
        return this->amplitude * std::sin(x + this->phase);
    }
};

TEST_CASE("compare_initX", "[analysis]") {
    const char *obs_key = "OBS1";
    const int ens_size = 100;
    const int obs_size = 10;
    std::vector<model> ens;
    bool_vector_type *ens_mask = bool_vector_alloc(ens_size, true);
    obs_data_type *obs_data = obs_data_alloc(1.0);
    meas_data_type *meas_data = meas_data_alloc(ens_mask);
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    model true_model{1.0, 0.0};

    for (int iens = 0; iens < ens_size; iens++)
        ens.emplace_back(rng);

    meas_block_type *mb = meas_data_add_block(meas_data, obs_key, 1, obs_size);
    obs_block_type *ob =
        obs_data_add_block(obs_data, obs_key, obs_size, nullptr, false);

    for (int iobs = 0; iobs < obs_size; iobs++) {
        double x = iobs * 2 * M_PI / (obs_size - 1);
        for (int iens = 0; iens < ens_size; iens++) {
            const auto &m = ens[iens];
            meas_block_iset(mb, iens, iobs, m.eval(x));
        }

        obs_block_iset(ob, iobs, true_model.eval(x), 0.25);
    }

    GIVEN("Simulation results") {
        matrix_type *S = meas_data_allocS(meas_data);
        matrix_type *R = obs_data_allocR(obs_data);
        matrix_type *E = obs_data_allocE(obs_data, rng, ens_size);
        matrix_type *D = obs_data_allocD(obs_data, E, S);

        WHEN("different truncations are used") {
            std_enkf_data_type *std_enkf_data =
                static_cast<std_enkf_data_type *>(std_enkf_data_alloc());

            matrix_type *X1 = matrix_alloc(ens_size, ens_size);
            matrix_type *X2 = matrix_alloc(ens_size, ens_size);
            std_enkf_set_truncation(std_enkf_data,
                                    GENERATE(DEFAULT_ENKF_TRUNCATION_, 0.90));

            std_enkf_initX(std_enkf_data, X1, nullptr, S, R, nullptr, E, D,
                           rng);
            ies::enkf_initX(std_enkf_get_truncation(std_enkf_data),
                            std_enkf_get_subspace_dimension(std_enkf_data),
                            std_enkf_data_get_inversion(std_enkf_data), S, R, E,
                            D, X2);

            REQUIRE(matrix_similar(X1, X2, 1e-10));
            std_enkf_data_free(std_enkf_data);
            matrix_free(X2);
            matrix_free(X1);
        }

        WHEN("Different inversion methods are used") {
            std_enkf_data_type *std_enkf_data =
                static_cast<std_enkf_data_type *>(std_enkf_data_alloc());

            matrix_type *X1 = matrix_alloc(ens_size, ens_size);
            matrix_type *X2 = matrix_alloc(ens_size, ens_size);
            std_enkf_set_string(std_enkf_data, INVERSION_KEY,
                                GENERATE(STRING_INVERSION_SUBSPACE_EE_R,
                                         STRING_INVERSION_SUBSPACE_EXACT_R,
                                         STRING_INVERSION_SUBSPACE_RE));

            std_enkf_initX(std_enkf_data, X1, nullptr, S, R, nullptr, E, D,
                           rng);
            ies::enkf_initX(std_enkf_get_truncation(std_enkf_data),
                            std_enkf_get_subspace_dimension(std_enkf_data),
                            std_enkf_data_get_inversion(std_enkf_data), S, R, E,
                            D, X2);

            REQUIRE(matrix_similar(X1, X2, 1e-10));
            std_enkf_data_free(std_enkf_data);
            matrix_free(X2);
            matrix_free(X1);
        }

        matrix_free(D);
        matrix_free(E);
        matrix_free(R);
        matrix_free(S);
    }

    obs_data_free(obs_data);
    meas_data_free(meas_data);
    rng_free(rng);
    bool_vector_free(ens_mask);
}
