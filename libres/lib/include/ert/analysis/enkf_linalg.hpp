#ifndef ERT_ENKF_LINALG_H
#define ERT_ENKF_LINALG_H
#include <variant>

#include <ert/util/double_vector.hpp>

#include <ert/res_util/matrix_lapack.hpp>
#include <ert/res_util/matrix.hpp>

void enkf_linalg_init_stdX(matrix_type *X, const matrix_type *S,
                           const matrix_type *D, const matrix_type *W,
                           const double *eig, bool bootstrap);

void enkf_linalg_init_sqrtX(matrix_type *X5, const matrix_type *S,
                            const matrix_type *randrot,
                            const matrix_type *innov, const matrix_type *W,
                            const double *eig, bool bootstrap);

void enkf_linalg_Cee(matrix_type *B, int nrens, const matrix_type *R,
                     const matrix_type *U0, const double *inv_sig0);

int enkf_linalg_svdS(const matrix_type *S,
                     const std::variant<double, int> &truncation,
                     dgesvd_vector_enum jobVT, double *sig0, matrix_type *U0,
                     matrix_type *V0T);

matrix_type *enkf_linalg_alloc_innov(const matrix_type *dObs,
                                     const matrix_type *S);

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
void enkf_linalg_genX3(matrix_type *X3, const matrix_type *W,
                       const matrix_type *D, const double *eig);

void enkf_linalg_meanX5(const matrix_type *S, const matrix_type *W,
                        const double *eig, const matrix_type *innov,
                        matrix_type *X5);

void enkf_linalg_X5sqrt(matrix_type *X2, matrix_type *X5,
                        const matrix_type *randrot, int nrobs);

matrix_type *enkf_linalg_alloc_mp_randrot(int ens_size, rng_type *rng);
void enkf_linalg_set_randrot(matrix_type *Q, rng_type *rng);
void enkf_linalg_checkX(const matrix_type *X, bool bootstrap);

#endif
