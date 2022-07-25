#include "catch2/catch.hpp"

#include "ert/analysis/enkf_linalg.hpp"

/*
Python equivalent:

W = np.array([[4, 3], [8, 6]])
D = np.array([[2, 3, 1], [4, 5, 2]])
eig = np.array([0.5, 0.3, 0.5])
ens_size = 2
Lambda = np.diag(eig[0:ens_size])

X3 = W @ Lambda @ W.T @ D
*/
Eigen::MatrixXd test_enkf_linalg_genX3() {
    // W has dimension min(ens_size, nrobs)
    Eigen::MatrixXd W{{4.0, 3.0}, {8.0, 6.0}};
    // D has dimension (nrobs, ens_size)
    Eigen::MatrixXd D{{2.0, 3.0, 1.0}, {4.0, 5.0, 2.0}};
    // eig has dimension (ens_size, 1)
    const Eigen::VectorXd eig{{0.5, 0.3, 0.5}};

    Eigen::MatrixXd X3 = enkf_linalg_genX3(W, D, eig);

    return X3;
}

TEST_CASE("enkf_linalg_genX3", "[analysis]") {
    Eigen::MatrixXd X3 = test_enkf_linalg_genX3();
    Eigen::MatrixXd result{{107.0, 139.1, 53.5}, {214.0, 278.2, 107.0}};
    REQUIRE(X3.isApprox(result, 1.0e-8));
}
