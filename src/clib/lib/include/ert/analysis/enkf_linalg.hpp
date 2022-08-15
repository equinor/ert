#ifndef ERT_ENKF_LINALG_H
#define ERT_ENKF_LINALG_H
#include <Eigen/Dense>
#include <variant>

int enkf_linalg_svdS(const Eigen::MatrixXd &S,
                     const std::variant<double, int> &truncation,
                     Eigen::VectorXd &inv_sig0, Eigen::MatrixXd &U0);

void enkf_linalg_lowrankCinv(
    const Eigen::MatrixXd &S, const Eigen::MatrixXd &R,
    Eigen::MatrixXd &W,   /* Corresponding to X1 from Eq. 14.29 */
    Eigen::VectorXd &eig, /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
    const std::variant<double, int> &truncation);

void enkf_linalg_lowrankE(
    const Eigen::MatrixXd &S, /* (nrobs x nrens) */
    const Eigen::MatrixXd &E, /* (nrobs x nrens) */
    Eigen::MatrixXd
        &W, /* (nrobs x nrmin) Corresponding to X1 from Eqs. 14.54-14.55 */
    Eigen::VectorXd
        &eig, /* (nrmin) Corresponding to 1 / (1 + Lambda1^2) (14.54) */
    const std::variant<double, int> &truncation);

Eigen::MatrixXd enkf_linalg_genX3(const Eigen::MatrixXd &W,
                                  const Eigen::MatrixXd &D,
                                  const Eigen::VectorXd &eig);
#endif
