/**
 * The tests in this file double both as unit testing
 * * */
#include <catch2/catch.hpp>

#include <ert/matrix.hpp>
#include <Eigen/Dense>


TEMPLATE_TEST_CASE("Matrix constructors", "[util]", ert::MatrixXd, Eigen::MatrixXd) {
    TestType empty_matrix;
    REQUIRE(empty_matrix.rows() == 0);
    REQUIRE(empty_matrix.cols() == 0);

    TestType initlist_matrix{{ 1, 2, 3 }, {4, 5, 6}};
    REQUIRE(initlist_matrix.rows() == 2);
    REQUIRE(initlist_matrix.cols() == 3);

    TestType copy_matrix = initlist_matrix;
    REQUIRE(copy_matrix.rows() == 2);
    REQUIRE(copy_matrix.cols() == 3);

    TestType copy_matrix2{initlist_matrix};
    REQUIRE(copy_matrix2.rows() == 2);
    REQUIRE(copy_matrix2.cols() == 3);

    TestType from_empty_matrix;
    from_empty_matrix = initlist_matrix;
    REQUIRE(from_empty_matrix.rows() == 2);
    REQUIRE(from_empty_matrix.cols() == 3);
}


TEMPLATE_TEST_CASE("Matrix arithmetic", "[util]", ert::MatrixXd, Eigen::MatrixXd) {
    GIVEN("two (n, m) non-empty matrices") {
        TestType mat_lhs{{ 1, 2, 3 }, {4, 5, 6}};
        TestType mat_rhs{{ 3, 2, 1 }, {2, 3, 4}};

        THEN("access coefficients") {
            REQUIRE(mat_lhs(0, 0) == 1);
            REQUIRE(mat_lhs(0, 1) == 2);
            REQUIRE(mat_lhs(0, 2) == 3);

            REQUIRE(mat_lhs(1, 0) == 4);
            REQUIRE(mat_lhs(1, 1) == 5);
            REQUIRE(mat_lhs(1, 2) == 6);
        }

        THEN("add") {
            auto mat = mat_lhs + mat_rhs;

            REQUIRE(mat(0, 0) == 4);
            REQUIRE(mat(0, 1) == 4);
            REQUIRE(mat(0, 2) == 4);

            REQUIRE(mat(1, 0) == 6);
            REQUIRE(mat(1, 1) == 8);
            REQUIRE(mat(1, 2) == 10);
        }

        THEN("in-place add") {
            mat_lhs += mat_rhs;

            REQUIRE(mat_lhs(0, 0) == 4);
            REQUIRE(mat_lhs(0, 1) == 4);
            REQUIRE(mat_lhs(0, 2) == 4);

            REQUIRE(mat_lhs(1, 0) == 6);
            REQUIRE(mat_lhs(1, 1) == 8);
            REQUIRE(mat_lhs(1, 2) == 10);
        }

        THEN("subtract") {
            auto mat = mat_lhs - mat_rhs;

            REQUIRE(mat(0, 0) == -2);
            REQUIRE(mat(0, 1) == 0);
            REQUIRE(mat(0, 2) == 2);

            REQUIRE(mat(1, 0) == 2);
            REQUIRE(mat(1, 1) == 2);
            REQUIRE(mat(1, 2) == 2);
        }

        THEN("in-place subtract") {
            mat_lhs -= mat_rhs;

            REQUIRE(mat_lhs(0, 0) == -2);
            REQUIRE(mat_lhs(0, 1) == 0);
            REQUIRE(mat_lhs(0, 2) == 2);

            REQUIRE(mat_lhs(1, 0) == 2);
            REQUIRE(mat_lhs(1, 1) == 2);
            REQUIRE(mat_lhs(1, 2) == 2);
        }

        THEN("transpose") {
            auto mat = mat_lhs.transpose();

            REQUIRE(mat(0, 0) == 1);
            REQUIRE(mat(0, 1) == 4);

            REQUIRE(mat(1, 0) == 2);
            REQUIRE(mat(1, 1) == 5);

            REQUIRE(mat(2, 0) == 3);
            REQUIRE(mat(2, 1) == 6);
        }

        THEN("matrix multiply") {
            auto mat = mat_lhs * mat_rhs.transpose();

            REQUIRE(mat(0, 0) == 10);
            REQUIRE(mat(0, 1) == 20);

            REQUIRE(mat(1, 0) == 28);
            REQUIRE(mat(1, 1) == 47);
        }

        THEN("in-place matrix multiply") {
            mat_lhs *= mat_rhs.transpose();

            REQUIRE(mat_lhs(0, 0) == 10);
            REQUIRE(mat_lhs(0, 1) == 20);

            REQUIRE(mat_lhs(1, 0) == 28);
            REQUIRE(mat_lhs(1, 1) == 47);
        }
    }
}
