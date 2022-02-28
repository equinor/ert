#include <algorithm>
#include <cmath>
#include <vector>

#include <stdlib.h>
#include <stdio.h>

#include <ert/res_util/matrix.hpp>
#include <ert/res_util/matrix_lapack.hpp>
#include <ert/res_util/matrix_blas.hpp>

#include <ert/analysis/enkf_linalg.hpp>

/**
 * Implements Eq. 14.31 in the book Data Assimilation,
 * The Ensemble Kalman Filter, 2nd Edition by Geir Evensen.
*/
void enkf_linalg_genX3(matrix_type *X3, const matrix_type *W,
                       const matrix_type *D, const double *eig) {
    const int nrobs = matrix_get_rows(D);
    const int nrens = matrix_get_columns(D);
    const int nrmin = std::min(nrobs, nrens);
    int i, j;
    matrix_type *X1 = matrix_alloc(nrmin, nrobs);
    matrix_type *X2 = matrix_alloc(nrmin, nrens);

    /* X1 = (I + Lambda1)^(-1) * W'*/
    for (i = 0; i < nrmin; i++)
        for (j = 0; j < nrobs; j++)
            matrix_iset(X1, i, j, eig[i] * matrix_iget(W, j, i));

    matrix_matmul(X2, X1, D); /*  X2 = X1 * D */
    matrix_matmul(X3, W, X2); /*  X3 = W * X2 = X1 * X2 */

    matrix_free(X1);
    matrix_free(X2);
}

/**
 * Implements a version of Eq. 14.34 from the book Data Assimilation,
 * The Ensemble Kalman Filter, 2nd Edition by Geir Evensen.
 *
 * Eq. 14.34 is as follows:
 *
 *     X2 = (I + \Lambda_1)^{-1/2} @ X^T_1 * S
 *
 * , but this function seems to implement the following:
 *
 *     X2 = X^T_1 * S
 *     X2 = (\Lambda^{-1/2}_1 * X2^T)^T
 *
 * See tests for more details.
*/
void enkf_linalg_genX2(matrix_type *X2, const matrix_type *S,
                       const matrix_type *W, const double *eig) {
    const int nrens = matrix_get_columns(S);
    const int idim = matrix_get_rows(X2);

    matrix_dgemm(X2, W, S, true, false, 1.0, 0.0);
    {
        int i, j;
        for (j = 0; j < nrens; j++)
            for (i = 0; i < idim; i++)
                matrix_imul(X2, i, j, sqrt(eig[i]));
    }
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
            /* Include one more singular value ? */
            if (running_sigma2 / total_sigma2 < truncation) {
                num_significant++;
                running_sigma2 += sig * sig;
            } else
                break;
        }
    }

    return num_significant;
}

int enkf_linalg_svdS(const matrix_type *S,
                     const std::variant<double, int> &truncation, Eigen::VectorXd &inv_sig0,
                     matrix_type *U0) {
    int num_significant = 0;

    auto svd = S->bdcSvd(Eigen::ComputeThinU);
    *U0 = svd.matrixU();

    if (std::holds_alternative<int>(truncation))
        num_significant = std::get<int>(truncation);
    else
        num_significant = enkf_linalg_num_significant(
            svd.singularValues(), std::get<double>(truncation));

    /* Inverting the significant singular values */
    inv_sig0 = svd.singularValues().cwiseInverse();
    inv_sig0.setZero(num_significant, 0, svd.singularValues())

    /* Explicitly setting the insignificant singular values to zero. */
    for (int i = num_significant; i < svd.singularValues().size(); i++)
        inv_sig0[i] = 0;

    return num_significant;
}

/*
 Routine computes X1 and eig corresponding to Eqs 14.54-14.55
 Geir Evensen
*/
void enkf_linalg_lowrankE(
    const matrix_type *S, /* (nrobs x nrens) */
    const matrix_type *E, /* (nrobs x nrens) */
    matrix_type
        *W, /* (nrobs x nrmin) Corresponding to X1 from Eqs. 14.54-14.55 */
    Eigen::VectorXd
        &eig, /* (nrmin)         Corresponding to 1 / (1 + Lambda1^2) (14.54) */
    const std::variant<double, int> &truncation) {

    const int nrobs = matrix_get_rows(S);
    const int nrens = matrix_get_columns(S);
    const int nrmin = std::min(nrobs, nrens);

    std::vector<double> inv_sig0(nrmin);
    matrix_type *U0 = matrix_alloc(nrobs, nrmin);
    matrix_type *X0 = matrix_alloc(nrmin, nrens);

    matrix_type *U1 = matrix_alloc(nrmin, nrmin);

    int i, j;

    /* Compute SVD of S=HA`  ->  U0, invsig0=sig0^(-1) */
    enkf_linalg_svdS(S, truncation, DGESVD_NONE, inv_sig0.data(), U0, NULL);

    /* X0(nrmin x nrens) =  Sigma0^(+) * U0'* E  (14.51)  */
    matrix_dgemm(X0, U0, E, true, false, 1.0,
                 0.0); /*  X0 = U0^T * E  (14.51) */

    /* Multiply X0 with sig0^(-1) from left X0 =  S^(-1) * X0   */
    for (j = 0; j < matrix_get_columns(X0); j++)
        for (i = 0; i < matrix_get_rows(X0); i++)
            matrix_imul(X0, i, j, inv_sig0[i]);

    /* Compute SVD of X0->  U1*eig*V1   14.52 */
    auto svd = X0->bdcSvd(Eigen::ComputeThinU);
    Eigen::VectorXd sig1 = svd.singularValues();
    *U1 = svd.matrixU();

    /* Lambda1 = 1/(I + Lambda^2)  in 14.56 */
    for (i = 0; i < nrmin; i++)
        eig[i] = 1.0 / (1.0 + sig1[i] * sig1[i]);

    /* Compute sig0^+ U1  (14:55) */
    for (j = 0; j < nrmin; j++)
        for (i = 0; i < nrmin; i++)
            matrix_imul(U1, i, j, inv_sig0[i]);

    /* Compute X1 = W = U0 * (U1=sig0^+ U1) = U0 * Sigma0^(+') * U1  (14:55) */
    matrix_matmul(W, U0, U1);

    matrix_free(X0);
    matrix_free(U0);
    matrix_free(U1);
}

void enkf_linalg_Cee(matrix_type *B, int nrens, const matrix_type *R,
                     const matrix_type *U0, const double *inv_sig0) {
    const int nrmin = matrix_get_rows(B);
    {
        matrix_type *X0 = matrix_alloc(nrmin, matrix_get_rows(R));
        matrix_dgemm(X0, U0, R, true, false, 1.0, 0.0);  /* X0 = U0^T * R */
        matrix_dgemm(B, X0, U0, false, false, 1.0, 0.0); /* B = X0 * U0 */
        matrix_free(X0);
    }

    {
        int i, j;

        /* Funny code ??
       Multiply B with S^(-1)from left and right
       BHat =  S^(-1) * B * S^(-1)
    */
        for (j = 0; j < matrix_get_columns(B); j++)
            for (i = 0; i < matrix_get_rows(B); i++)
                matrix_imul(B, i, j, inv_sig0[i]);

        for (j = 0; j < matrix_get_columns(B); j++)
            for (i = 0; i < matrix_get_rows(B); i++)
                matrix_imul(B, i, j, inv_sig0[j]);
    }

    matrix_scale(B, nrens - 1.0);
}

void enkf_linalg_lowrankCinv__(const matrix_type *S, const matrix_type *R, matrix_type *Z,
                               Eigen::VectorXd &eig, matrix_type *U0,
                               const std::variant<double, int> &truncation) {

    const int nrobs = matrix_get_rows(S);
    const int nrens = matrix_get_columns(S);
    const int nrmin = std::min(nrobs, nrens);
    std::vector<double> inv_sig0(nrmin);

    enkf_linalg_svdS(S, truncation, inv_sig0.data(), U0);

    {
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nrmin, nrmin);
        enkf_linalg_Cee(
            &B, nrens, R, U0,
            inv_sig0
                .data()); /* B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/
        auto svd = B.bdcSvd(Eigen::ComputeThinU);
        *Z = svd.matrixU();
        eig = svd.singularValues();
    }

    {
        int i, j;
        /* Lambda1 = (I + Lambda)^(-1) */

        for (i = 0; i < nrmin; i++)
            eig[i] = 1.0 / (1 + eig[i]);

        for (j = 0; j < nrmin; j++)
            for (i = 0; i < nrmin; i++)
                matrix_imul(Z, i, j, inv_sig0[i]); /* Z2 =  Sigma0^(+) * Z; */
    }
}

void enkf_linalg_lowrankCinv(
    const matrix_type *S, const matrix_type *R,
    matrix_type *W,       /* Corresponding to X1 from Eq. 14.29 */
    Eigen::VectorXd &eig, /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
    const std::variant<double, int> &truncation) {

    const int nrobs = matrix_get_rows(S);
    const int nrens = matrix_get_columns(S);
    const int nrmin = std::min(nrobs, nrens);

    matrix_type *U0 = matrix_alloc(nrobs, nrmin);
    matrix_type *Z = matrix_alloc(nrmin, nrmin);

    enkf_linalg_lowrankCinv__(S, R, Z, eig, U0, truncation);
    matrix_matmul(W, U0, Z); /* X1 = W = U0 * Z2 = U0 * Sigma0^(+') * Z    */

    matrix_free(U0);
    matrix_free(Z);
}
