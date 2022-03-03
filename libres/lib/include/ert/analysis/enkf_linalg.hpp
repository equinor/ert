#ifndef ERT_ENKF_LINALG_H
#define ERT_ENKF_LINALG_H
#include <variant>

#include <ert/util/double_vector.hpp>

#include <ert/res_util/matrix_lapack.hpp>
#include <ert/res_util/matrix.hpp>

void enkf_linalg_Cee(matrix_type *B, int nrens, const matrix_type *R,
                     const matrix_type *U0, const double *inv_sig0);

int enkf_linalg_svdS(const matrix_type *S,
                     const std::variant<double, int> &truncation,
                     dgesvd_vector_enum jobVT, double *sig0, matrix_type *U0,
                     matrix_type *V0T);

void enkf_linalg_lowrankCinv__(const matrix_type *S, const matrix_type *R,
                               matrix_type *V0T, matrix_type *Z, double *eig,
                               matrix_type *U0,
                               const std::variant<double, int> &truncation);

void enkf_linalg_lowrankCinv(
    const matrix_type *S, const matrix_type *R,
    matrix_type *W, /* Corresponding to X1 from Eq. 14.29 */
    double *eig,    /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
    const std::variant<double, int> &truncation);

void enkf_linalg_lowrankE(
    const matrix_type *S, /* (nrobs x nrens) */
    const matrix_type *E, /* (nrobs x nrens) */
    matrix_type
        *W, /* (nrobs x nrmin) Corresponding to X1 from Eqs. 14.54-14.55 */
    double
        *eig, /* (nrmin)         Corresponding to 1 / (1 + Lambda1^2) (14.54) */
    const std::variant<double, int> &truncation);

void enkf_linalg_genX2(matrix_type *X2, const matrix_type *S,
                       const matrix_type *W, const double *eig);

Eigen::MatrixXd enkf_linalg_genX3(const Eigen::MatrixXd &W,
                                  const Eigen::MatrixXd &D,
                                  const Eigen::VectorXd &eig);
#endif
