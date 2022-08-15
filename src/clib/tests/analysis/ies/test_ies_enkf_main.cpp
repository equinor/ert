#include <vector>

#include <Eigen/Dense>
#include <catch2/catch.hpp>
#include <ert/util/rng.h>
#include <ert/util/util.h>

#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_data.hpp>

namespace ies {
void linalg_exact_inversion(Eigen::MatrixXd &W0, const Eigen::MatrixXd &S,
                            const Eigen::MatrixXd &H, double ies_steplength);
} // namespace ies

TEST_CASE("make_activeE", "[analysis]") {
    int obs_size = 3;
    int ens_size = 2;

    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    ies::Data data(ens_size);

    // Initialising masks such that all observations and realizations are active
    std::vector<bool> ens_mask(ens_size, true);
    data.update_ens_mask(ens_mask);

    std::vector<bool> obs_mask(obs_size, true);
    data.store_initial_obs_mask(obs_mask);
    data.update_obs_mask(obs_mask);

    // Set initial data
    Eigen::MatrixXd Ein = Eigen::MatrixXd::Zero(obs_size, ens_size);

    // Set first column
    Ein(0, 0) = 1.0;
    Ein(1, 0) = 2.0;
    Ein(2, 0) = 3.0;

    // Set second column
    Ein(0, 1) = 1.5;
    Ein(1, 1) = 2.5;
    Ein(2, 1) = 3.5;
    data.store_initialE(Ein);

    SECTION("make_activeE() does nothing when all "
            "observations and realizations are active") {
        auto E = data.make_activeE();
        REQUIRE(Ein == E);
    }

    SECTION("deactivate one realisation") {
        ens_mask[1] = false;
        data.update_ens_mask(ens_mask);

        auto E = data.make_activeE();
        REQUIRE(E.rows() == 3);
        REQUIRE(E.cols() == 1);

        REQUIRE(E(0, 0) == 1.0);
        REQUIRE(E(1, 0) == 2.0);
        REQUIRE(E(2, 0) == 3.0);
    }

    SECTION("deactivate one observation") {
        obs_mask[1] = false;
        data.update_obs_mask(obs_mask);

        auto E = data.make_activeE();
        REQUIRE(E.rows() == 2);
        REQUIRE(E.cols() == 2);

        REQUIRE(E(0, 0) == 1.0);
        REQUIRE(E(1, 0) == 3.0);
        REQUIRE(E(0, 1) == 1.5);
        REQUIRE(E(1, 1) == 3.5);
    }

    SECTION("deactivate one observation and one realisation") {
        obs_mask[1] = false;
        data.update_obs_mask(obs_mask);

        ens_mask[1] = false;
        data.update_ens_mask(ens_mask);

        auto E = data.make_activeE();
        REQUIRE(E.rows() == 2);
        REQUIRE(E.cols() == 1);

        REQUIRE(E(0, 0) == 1.0);
        REQUIRE(E(1, 0) == 3.0);
    }

    rng_free(rng);
}

TEST_CASE("make_activeW", "[analysis]") {
    const int ens_size = 4;
    const int obs_size = 10;
    ies::Data data(ens_size);
    std::vector<bool> ens_mask(ens_size, true);
    std::vector<bool> obs_mask(obs_size, true);

    ies::init_update(data, ens_mask, obs_mask);
    data.update_ens_mask(ens_mask);

    Eigen::MatrixXd W0 = Eigen::MatrixXd::Zero(ens_size, ens_size);
    for (int i = 0; i < ens_size; i++) {
        for (int j = 0; j < ens_size; j++)
            W0(i, j) = i * ens_size + j;
    }
    ies::linalg_store_active_W(data, W0);

    {
        auto W = data.make_activeW();
        REQUIRE(W == W0);
    }

    // Deactivate one realization
    ens_mask[1] = false;
    data.update_ens_mask(ens_mask);
    {
        auto W = data.make_activeW();
        for (int i = 0; i < ens_size - 1; i++) {
            for (int j = 0; j < ens_size - 1; j++) {
                int i0 = i + (i > 0);
                int j0 = j + (j > 0);
                REQUIRE(W(i, j) == W0(i0, j0));
            }
        }
    }
}

SCENARIO("make_activeA", "[analysis]") {
    GIVEN("Inital setup") {
        const int ens_size = 4;
        const int obs_size = 10;
        const int state_size = 10;
        ies::Data data(ens_size);
        std::vector<bool> ens_mask(ens_size, true);
        std::vector<bool> obs_mask(obs_size, true);
        Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(state_size, ens_size);
        for (int i = 0; i < state_size; i++) {
            for (int j = 0; j < ens_size; j++)
                A0(i, j) = i * ens_size + j;
        }
        ies::init_update(data, ens_mask, obs_mask);
        data.store_initialA(A0);

        WHEN("All realizations active") {
            auto A = data.make_activeA();
            REQUIRE(A == A0);
        }

        WHEN("One realization deactivated") {
            int dead_iens = 2;
            ens_mask[dead_iens] = false;
            data.update_ens_mask(ens_mask);
            auto A = data.make_activeA();
            for (int i = 0; i < state_size; i++) {
                int i0 = i;
                for (int j = 0; j < ens_size - 1; j++) {
                    int j0 = j;
                    if (j0 >= dead_iens)
                        j0 += 1;

                    REQUIRE(A(i, j) == A0(i0, j0));
                }
            }
        }
    }
}

TEST_CASE("linalg_exact_inversion", "[analysis]") {
    int ens_size = 5;
    int obs_size = 3;

    Eigen::MatrixXd W0{{1, 2, 3, 4, 5},
                       {6, 7, 8, 9, 10},
                       {11, 12, 13, 14, 15},
                       {16, 17, 18, 19, 20},
                       {21, 22, 23, 24, 25}};
    Eigen::MatrixXd S = Eigen::MatrixXd::Constant(obs_size, ens_size, 1.0);
    Eigen::MatrixXd H = Eigen::MatrixXd::Constant(obs_size, ens_size, 1.0);
    double ies_steplength = 0.5;

    ies::linalg_exact_inversion(W0, S, H, ies_steplength);

    Eigen::MatrixXd Expected{{0.59375, 1.09375, 1.59375, 2.09375, 2.59375},
                             {3.09375, 3.59375, 4.09375, 4.59375, 5.09375},
                             {5.59375, 6.09375, 6.59375, 7.09375, 7.59375},
                             {8.09375, 8.59375, 9.09375, 9.59375, 10.0938},
                             {10.5938, 11.0938, 11.5938, 12.0938, 12.5938}};

    REQUIRE(W0.isApprox(Expected, 1e-5));
}
