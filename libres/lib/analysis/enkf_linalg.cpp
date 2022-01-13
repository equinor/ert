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

static int enkf_linalg_num_significant(int num_singular_values,
                                       const double *sig0, double truncation) {
    int num_significant = 0;
    double total_sigma2 = 0;
    for (int i = 0; i < num_singular_values; i++)
        total_sigma2 += sig0[i] * sig0[i];

    /*
     * Determine the number of singular values by enforcing that
     * less than a fraction @truncation of the total variance be
     * accounted for.
     */
    {
        double running_sigma2 = 0;
        for (int i = 0; i < num_singular_values; i++) {
            if (running_sigma2 / total_sigma2 <
                truncation) { /* Include one more singular value ? */
                num_significant++;
                running_sigma2 += sig0[i] * sig0[i];
            } else
                break;
        }
    }

    return num_significant;
}

int enkf_linalg_svdS(const matrix_type *S,
                     const std::variant<double, int> &truncation,
                     dgesvd_vector_enum store_V0T, double *inv_sig0,
                     matrix_type *U0, matrix_type *V0T) {

    double *sig0 = inv_sig0;
    int num_significant = 0;

    int num_singular_values =
        std::min(matrix_get_rows(S), matrix_get_columns(S));
    {
        matrix_type *workS = matrix_alloc_copy(S);
        matrix_dgesvd(DGESVD_MIN_RETURN, store_V0T, workS, sig0, U0, V0T);
        matrix_free(workS);
    }

    if (std::holds_alternative<int>(truncation))
        num_significant = std::get<int>(truncation);
    else
        num_significant = enkf_linalg_num_significant(
            num_singular_values, sig0, std::get<double>(truncation));

    {
        int i;
        /* Inverting the significant singular values */
        for (i = 0; i < num_significant; i++)
            inv_sig0[i] = 1.0 / sig0[i];

        /* Explicitly setting the insignificant singular values to zero. */
        for (i = num_significant; i < num_singular_values; i++)
            inv_sig0[i] = 0;
    }

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
    double
        *eig, /* (nrmin)         Corresponding to 1 / (1 + Lambda1^2) (14.54) */
    const std::variant<double, int> &truncation) {

    const int nrobs = matrix_get_rows(S);
    const int nrens = matrix_get_columns(S);
    const int nrmin = std::min(nrobs, nrens);

    std::vector<double> inv_sig0(nrmin);
    matrix_type *U0 = matrix_alloc(nrobs, nrmin);
    matrix_type *X0 = matrix_alloc(nrmin, nrens);

    matrix_type *U1 = matrix_alloc(nrmin, nrmin);
    std::vector<double> sig1(nrmin);

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
    matrix_dgesvd(DGESVD_MIN_RETURN, DGESVD_NONE, X0, sig1.data(), U1, NULL);

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

void enkf_linalg_lowrankCinv__(const matrix_type *S, const matrix_type *R,
                               matrix_type *V0T, matrix_type *Z, double *eig,
                               matrix_type *U0,
                               const std::variant<double, int> &truncation) {

    const int nrobs = matrix_get_rows(S);
    const int nrens = matrix_get_columns(S);
    const int nrmin = std::min(nrobs, nrens);
    std::vector<double> inv_sig0(nrmin);

    if (V0T != NULL)
        enkf_linalg_svdS(S, truncation, DGESVD_MIN_RETURN, inv_sig0.data(), U0,
                         V0T);
    else
        enkf_linalg_svdS(S, truncation, DGESVD_NONE, inv_sig0.data(), U0, NULL);

    {
        matrix_type *B = matrix_alloc(nrmin, nrmin);
        enkf_linalg_Cee(
            B, nrens, R, U0,
            inv_sig0
                .data()); /* B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/
        matrix_dgesvd(DGESVD_MIN_RETURN, DGESVD_NONE, B, eig, Z, NULL);
        matrix_free(B);
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
    matrix_type *W, /* Corresponding to X1 from Eq. 14.29 */
    double *eig,    /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
    const std::variant<double, int> &truncation) {

    const int nrobs = matrix_get_rows(S);
    const int nrens = matrix_get_columns(S);
    const int nrmin = std::min(nrobs, nrens);

    matrix_type *U0 = matrix_alloc(nrobs, nrmin);
    matrix_type *Z = matrix_alloc(nrmin, nrmin);

    enkf_linalg_lowrankCinv__(S, R, NULL, Z, eig, U0, truncation);
    matrix_matmul(W, U0, Z); /* X1 = W = U0 * Z2 = U0 * Sigma0^(+') * Z    */

    matrix_free(U0);
    matrix_free(Z);
}

void enkf_linalg_meanX5(const matrix_type *S, const matrix_type *W,
                        const double *eig, const matrix_type *dObs,
                        matrix_type *X5) {

    const int nrens = matrix_get_columns(S);
    const int nrobs = matrix_get_rows(S);
    const int nrmin = std::min(nrobs, nrens);
    std::vector<double> work(2 * nrmin + nrobs + nrens);
    matrix_type *innov = enkf_linalg_alloc_innov(dObs, S);
    {
        double *y1 = &work[0];
        double *y2 = &work[nrmin];
        double *y3 = &work[2 * nrmin];
        double *y4 = &work[2 * nrmin + nrobs];

        if (nrobs == 1) {
            /* Is this special casing necessary ??? */
            y1[0] = matrix_iget(W, 0, 0) * matrix_iget(innov, 0, 0);
            y2[0] = eig[0] * y1[0];
            y3[0] = matrix_iget(W, 0, 0) * y2[0];
            for (int iens = 0; iens < nrens; iens++)
                y4[iens] = y3[0] * matrix_iget(S, 0, iens);
        } else {
            matrix_dgemv(W, matrix_get_data(innov), y1, true, 1.0,
                         0.0); /* y1 = Trans(W) * innov */
            for (int i = 0; i < nrmin; i++)
                y2[i] = eig[i] * y1[i];               /* y2 = eig * y1      */
            matrix_dgemv(W, y2, y3, false, 1.0, 0.0); /* y3 = W * y2;       */
            matrix_dgemv(S, y3, y4, true, 1.0, 0.0);  /* y4 = Trans(S) * y3 */
        }

        for (int iens = 0; iens < nrens; iens++)
            matrix_set_column(X5, y4, iens);

        matrix_shift(X5, 1.0 / nrens);
    }
    matrix_free(innov);
}

void enkf_linalg_X5sqrt(matrix_type *X2, matrix_type *X5,
                        const matrix_type *randrot, int nrobs) {
    const int nrens = matrix_get_columns(X5);
    const int nrmin = std::min(nrobs, nrens);
    matrix_type *VT = matrix_alloc(nrens, nrens);
    std::vector<double> sig(nrmin);
    std::vector<double> isig(nrmin);

    matrix_dgesvd(DGESVD_NONE, DGESVD_ALL, X2, sig.data(), NULL, VT);
    {
        matrix_type *X3 = matrix_alloc(nrens, nrens);
        matrix_type *X33 = matrix_alloc(nrens, nrens);
        matrix_type *X4 = matrix_alloc(nrens, nrens);
        matrix_type *IenN = matrix_alloc(nrens, nrens);
        int i, j;
        for (i = 0; i < nrmin; i++)
            isig[i] = sqrt(util_double_max(1.0 - sig[i] * sig[i], 0.0));

        for (j = 0; j < nrens; j++)
            for (i = 0; i < nrens; i++)
                matrix_iset(X3, i, j, matrix_iget(VT, j, i));

        for (j = 0; j < nrmin; j++)
            matrix_scale_column(X3, j, isig[j]);

        matrix_dgemm(X33, X3, VT, false, false, 1.0, 0.0); /* X33 = X3   * VT */
        if (randrot != NULL)
            matrix_dgemm(X4, X33, randrot, false, false, 1.0,
                         0.0); /* X4  = X33  * Randrot */
        else
            matrix_assign(X4, X33);

        matrix_set(IenN, -1.0 / nrens);
        for (i = 0; i < nrens; i++)
            matrix_iadd(IenN, i, i, 1.0);

        matrix_dgemm(X5, IenN, X4, false, false, 1.0,
                     1.0); /* X5  = IenN * X4 + X5 */

        matrix_free(X3);
        matrix_free(X33);
        matrix_free(X4);
        matrix_free(IenN);
    }
    matrix_free(VT);
}

matrix_type *enkf_linalg_alloc_innov(const matrix_type *dObs,
                                     const matrix_type *S) {
    matrix_type *innov = matrix_alloc_copy(dObs);

    for (int iobs = 0; iobs < matrix_get_row_sum(dObs, iobs); iobs++)
        matrix_isub(innov, iobs, 0, matrix_get_row_sum(S, iobs));

    return innov;
}

void enkf_linalg_init_stdX(matrix_type *X, const matrix_type *S,
                           const matrix_type *D, const matrix_type *W,
                           const double *eig, bool bootstrap) {

    int nrobs = matrix_get_rows(W);
    int ens_size = matrix_get_rows(X);

    matrix_type *X3 = matrix_alloc(nrobs, ens_size);
    enkf_linalg_genX3(
        X3, W, D,
        eig); /*  X2 = diag(eig) * W' * D (Eq. 14.31, Evensen (2007)) */
              /*  X3 = W * X2 = X1 * X2 (Eq. 14.31, Evensen (2007)) */

    matrix_dgemm(X, S, X3, true, false, 1.0, 0.0); /* X = S' * X3 */
    if (!bootstrap) {
        for (int i = 0; i < ens_size; i++)
            matrix_iadd(X, i, i, 1.0); /*X = I + X */
    }

    matrix_free(X3);
}

void enkf_linalg_init_sqrtX(matrix_type *X5, const matrix_type *S,
                            const matrix_type *randrot,
                            const matrix_type *innov, const matrix_type *W,
                            const double *eig, bool bootstrap) {

    const int nrobs = matrix_get_rows(S);
    const int nrens = matrix_get_columns(S);
    const int nrmin = std::min(nrobs, nrens);

    matrix_type *X2 = matrix_alloc(nrmin, nrens);

    if (bootstrap)
        util_exit("%s: Sorry bootstrap support not fully implemented for SQRT "
                  "scheme\n",
                  __func__);

    enkf_linalg_meanX5(S, W, eig, innov, X5);
    enkf_linalg_genX2(X2, S, W, eig);
    enkf_linalg_X5sqrt(X2, X5, randrot, nrobs);

    matrix_free(X2);
}

/*
   This routine generates a real orthogonal random matrix.
   The algorithm is the one by
   Francesco Mezzadri (2007), How to generate random matrices from the classical
   compact groups, Notices of the AMS, Vol. 54, pp 592-604.
   1. First a matrix with independent random normal numbers are simulated.
   2. Then the QR decomposition is computed, and Q will then be a random orthogonal matrix.
   3. The diagonal elements of R are extracted and we construct the diagonal matrix X(j,j)=R(j,j)/|R(j,j)|
   4. An updated Q'=Q X is computed, and this is now a random orthogonal matrix with a Haar measure.

   The implementation is a plain reimplementation/copy of the old m_randrot.f90 function.
*/
void enkf_linalg_set_randrot(matrix_type *Q, rng_type *rng) {
    int ens_size = matrix_get_rows(Q);
    std::vector<double> tau(ens_size);
    std::vector<int> sign(ens_size);

    for (int i = 0; i < ens_size; i++)
        for (int j = 0; j < ens_size; j++)
            matrix_iset(Q, i, j, rng_std_normal(rng));

    matrix_dgeqrf(Q, tau.data()); /* QR factorization */
    for (int i = 0; i < ens_size; i++) {
        double Qii = matrix_iget(Q, i, i);
        sign[i] = (Qii > 0) ? 1 : -1;
    }

    matrix_dorgqr(Q, tau.data(), ens_size);
    for (int i = 0; i < ens_size; i++) {
        if (sign[i] < 0)
            matrix_scale_column(Q, i, -1);
    }
}

/*
   Generates the mean preserving random rotation for the EnKF SQRT algorithm
   using the algorithm from Sakov 2006-07.  I.e, generate rotation Up such that
   Up*Up^T=I and Up*1=1 (all rows have sum = 1)  see eq 17.
   From eq 18,    Up=B * Upb * B^T
   B is a random orthonormal basis with the elements in the first column equals 1/sqrt(nrens)

   Upb = | 1  0 |
   | 0  U |

   where U is an arbitrary orthonormal matrix of dim nrens-1 x nrens-1  (eq. 19)
*/

matrix_type *enkf_linalg_alloc_mp_randrot(int ens_size, rng_type *rng) {
    matrix_type *Up = matrix_alloc(ens_size, ens_size); /* The return value. */
    {
        matrix_type *B = matrix_alloc(ens_size, ens_size);
        matrix_type *Upb = matrix_alloc(ens_size, ens_size);
        matrix_type *U =
            matrix_alloc_shared(Upb, 1, 1, ens_size - 1, ens_size - 1);

        {
            int k, j;
            matrix_type *R = matrix_alloc(ens_size, ens_size);
            matrix_random_init(B,
                               rng); /* B is filled up with U(0,1) numbers. */
            matrix_set_const_column(B, 1.0 / sqrt(ens_size), 0);

            /* modified_gram_schmidt is used to create the orthonormal basis in B.*/
            for (k = 0; k < ens_size; k++) {
                double Rkk = sqrt(matrix_column_column_dot_product(B, k, B, k));
                matrix_iset(R, k, k, Rkk);
                matrix_scale_column(B, k, 1.0 / Rkk);
                for (j = k + 1; j < ens_size; j++) {
                    double Rkj = matrix_column_column_dot_product(B, k, B, j);
                    matrix_iset(R, k, j, Rkj);
                    {
                        int i;
                        for (i = 0; i < ens_size; i++) {
                            double Bij = matrix_iget(B, i, j);
                            double Bik = matrix_iget(B, i, k);
                            matrix_iset(B, i, j, Bij - Bik * Rkj);
                        }
                    }
                }
            }
            matrix_free(R);
        }

        enkf_linalg_set_randrot(U, rng);
        matrix_iset(Upb, 0, 0, 1);

        {
            matrix_type *Q = matrix_alloc(ens_size, ens_size);
            matrix_dgemm(Q, B, Upb, false, false, 1, 0); /* Q  = B * Ubp  */
            matrix_dgemm(Up, Q, B, false, true, 1, 0);   /* Up = Q * T(B) */
            matrix_free(Q);
        }

        matrix_free(B);
        matrix_free(Upb);
        matrix_free(U);
    }

    return Up;
}

/*
   Checking that the sum through one row in the X matrix equals
   @target_sum. @target_sum will be 1 normally, and zero if we are doing
   bootstrap.
*/
void enkf_linalg_checkX(const matrix_type *X, bool bootstrap) {
    matrix_assert_finite(X);
    {
        int target_sum;
        if (bootstrap)
            target_sum = 0;
        else
            target_sum = 1;

        for (int icol = 0; icol < matrix_get_columns(X); icol++) {
            double col_sum = matrix_get_column_sum(X, icol);
            if (std::abs(col_sum - target_sum) > 0.0001)
                util_abort("%s: something is seriously broken. col:%d  col_sum "
                           "= %g != %g - ABORTING\n",
                           __func__, icol, col_sum, target_sum);
        }
    }
}
