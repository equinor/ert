/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'matrix_lapack.h' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/
#ifndef ERT_MATRIX_LAPACK_H
#define ERT_MATRIX_LAPACK_H

#include <ert/res_util/matrix.hpp>

typedef enum {
    /* A */ DSYEVX_ALL, /* Compute all the eigenvalues */
                        /* V */
    DSYEVX_VALUE_INTERVAL, /* Computes eigenvalues in half open interval <VL , VU] */
    /* I */ DSYEVX_INDEX_INTERVAL
} /* The IL-th through IU'th eigenvalue will be found. */
dsyevx_eig_enum;

typedef enum {
    /* U */
    DSYEVX_AUPPER, /* The data is stored in the upper right part of the A matrix. */
    /* L */ DSYEVX_ALOWER
} /* The data is stored in the lower left  part of the A matrix. */
dsyevx_uplo_enum;

void matrix_dgesv(matrix_type *A, matrix_type *B);
void matrix_dgesvx(matrix_type *A, matrix_type *B, double *rcond);

int matrix_dsyevx(bool compute_eig_vectors, dsyevx_eig_enum which_values,
                  dsyevx_uplo_enum uplo, matrix_type *A, double VL, double VU,
                  int IL, int IU, double *eig_values, matrix_type *Z);

void matrix_dgeqrf(matrix_type *A, double *tau);
void matrix_dorgqr(matrix_type *A, double *tau, int num_reflectors);

double matrix_det(matrix_type *A);

int matrix_inv(matrix_type *A);

#endif
