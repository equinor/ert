#include <Eigen/Dense>
#include <catch2/catch.hpp>

#include <ert/analysis/ies/ies.hpp>

TEST_CASE("ies_make_D", "[analysis]") {
    GIVEN("An E matrix, an S matrix and a obeservation vector") {
        Eigen::MatrixXd S(2, 2);
        S << 2.0, 4.0, 6.0, 8.0;
        Eigen::MatrixXd E(2, 2);
        E << 1.0, 2.0, 3.0, 4.0;
        Eigen::VectorXd obs_values(2);
        obs_values << 1.0, 1.0;
        THEN("Generating D") {
            auto D = ies::makeD(obs_values, E, S);
            Eigen::MatrixXd D_expected(2, 2);
            D_expected << 1.0 - 2 + 1.0, 2.0 - 4.0 + 1.0, 3.0 - 6.0 + 1.0,
                4.0 - 8 + 1.0;
            REQUIRE(D == D_expected);
        }
    }
}
