#include <algorithm>
#include <cmath>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

#include <ert/analysis/enkf_linalg.hpp>

/**
 * Implements parts of Eq. 14.31 in the book Data Assimilation,
 * The Ensemble Kalman Filter, 2nd Edition by Geir Evensen.
 * Specifically, this implements
 * X_1 (I + \Lambda_1)^{-1} X_1^T (D - M[A^f])
*/
Eigen::MatrixXd enkf_linalg_genX3(const Eigen::MatrixXd &W,
                                  const Eigen::MatrixXd &D,
                                  const Eigen::VectorXd &eig) {
    const int nrmin = std::min(D.rows(), D.cols());
    // Corresponds to (I + \Lambda_1)^{-1} since `eig` has already been transformed.
    Eigen::MatrixXd Lambda_inv = eig(Eigen::seq(0, nrmin - 1)).asDiagonal();
    Eigen::MatrixXd X1 = Lambda_inv * W.transpose();

    Eigen::MatrixXd X2 = X1 * D;
    Eigen::MatrixXd X3 = W * X2;

    return X3;
}

static int enkf_linalg_num_significant(const Eigen::VectorXd &singular_values,
                                       double truncation) {
    int num_significant = 0;
    double total_sigma2 = singular_values.squaredNorm();

    /*
     * Determine the number of singular values by enforcing that
     * less than a fraction @truncation of the total variance be
     * accounted for.
     */
    {
        double running_sigma2 = 0;
        for (auto sig : singular_values) {
            if (running_sigma2 / total_sigma2 <
                truncation) { /* Include one more singular value ? */
                num_significant++;
                running_sigma2 += sig * sig;
            } else
                break;
        }
    }

    return num_significant;
}

int enkf_linalg_svdS(const Eigen::MatrixXd &S,
                     const std::variant<double, int> &truncation,
                     Eigen::VectorXd &inv_sig0, Eigen::MatrixXd &U0) {

    int num_significant = 0;

    auto svd = S.bdcSvd(Eigen::ComputeThinU);
    U0 = svd.matrixU();
    Eigen::VectorXd singular_values = svd.singularValues();

    if (std::holds_alternative<int>(truncation)) {
        num_significant = std::get<int>(truncation);
    } else {
        num_significant = enkf_linalg_num_significant(
            singular_values, std::get<double>(truncation));
    }

    inv_sig0 = singular_values.cwiseInverse();

    inv_sig0(Eigen::seq(num_significant, Eigen::last)).setZero();

    return num_significant;
}

/**
 Routine computes X1 and eig corresponding to Eqs 14.54-14.55
 Geir Evensen
*/
void enkf_linalg_lowrankE(
    const Eigen::MatrixXd &S, /* (nrobs x nrens) */
    const Eigen::MatrixXd &E, /* (nrobs x nrens) */
    Eigen::MatrixXd
        &W, /* (nrobs x nrmin) Corresponding to X1 from Eqs. 14.54-14.55 */
    Eigen::VectorXd
        &eig, /* (nrmin)         Corresponding to 1 / (1 + Lambda1^2) (14.54) */
    const std::variant<double, int> &truncation) {

    const int nrobs = S.rows();
    const int nrens = S.cols();
    const int nrmin = std::min(nrobs, nrens);

    Eigen::VectorXd inv_sig0(nrmin);
    Eigen::MatrixXd U0(nrobs, nrmin);

    /* Compute SVD of S=HA`  ->  U0, invsig0=sig0^(-1) */
    enkf_linalg_svdS(S, truncation, inv_sig0, U0);

    Eigen::MatrixXd Sigma_inv = inv_sig0.asDiagonal();

    /* X0(nrmin x nrens) =  Sigma0^(+) * U0'* E  (14.51)  */
    Eigen::MatrixXd X0 = Sigma_inv * U0.transpose() * E;

    /* Compute SVD of X0->  U1*eig*V1   14.52 */
    auto svd = X0.bdcSvd(Eigen::ComputeThinU);
    const auto &sig1 = svd.singularValues();

    /* Lambda1 = 1/(I + Lambda^2)  in 14.56 */
    for (int i = 0; i < nrmin; i++)
        eig[i] = 1.0 / (1.0 + sig1[i] * sig1[i]);

    /* Compute X1 = W = U0 * (U1=sig0^+ U1) = U0 * Sigma0^(+') * U1  (14.55) */
    W = U0 * Sigma_inv.transpose() * svd.matrixU();
}

void enkf_linalg_lowrankCinv(
    const Eigen::MatrixXd &S, const Eigen::MatrixXd &R,
    Eigen::MatrixXd &W,   /* Corresponding to X1 from Eq. 14.29 */
    Eigen::VectorXd &eig, /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
    const std::variant<double, int> &truncation) {

    const int nrobs = S.rows();
    const int nrens = S.cols();
    const int nrmin = std::min(nrobs, nrens);

    Eigen::MatrixXd U0(nrobs, nrmin);
    Eigen::MatrixXd Z(nrmin, nrmin);

    Eigen::VectorXd inv_sig0(nrmin);
    enkf_linalg_svdS(S, truncation, inv_sig0, U0);

    Eigen::MatrixXd Sigma_inv = inv_sig0.asDiagonal();

    /* B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/
    Eigen::MatrixXd B = (nrens - 1.0) * Sigma_inv * U0.transpose() * R * U0 *
                        Sigma_inv.transpose();

    auto svd = B.bdcSvd(Eigen::ComputeThinU);
    Z = svd.matrixU();
    eig = svd.singularValues();

    /* Lambda1 = (I + Lambda)^(-1) */
    for (int i = 0; i < nrmin; i++)
        eig[i] = 1.0 / (1 + eig[i]);

    Z = Sigma_inv * Z;

    W = U0 * Z; /* X1 = W = U0 * Z2 = U0 * Sigma0^(+') * Z    */
}
