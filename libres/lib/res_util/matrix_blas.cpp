/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'matrix_blas.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/util.hpp>

#include <ert/res_util/matrix.hpp>
#include <ert/res_util/matrix_blas.hpp>

/*
   C = alpha * op(A) * op(B)  +  beta * C

   op(·) can either be unity or Transpose.
*/

void matrix_dgemm(matrix_type *C, const matrix_type *A, const matrix_type *B,
                  bool transA, bool transB, double alpha, double beta) {
    Eigen::MatrixXd matrix_a;
    if (transA) {
        matrix_a = A->transpose();
    } else {
        matrix_a = *A;
    }

    Eigen::MatrixXd matrix_b;
    if (transB) {
        matrix_b = B->transpose();
    } else {
        matrix_b = *B;
    }

    Eigen::MatrixXd tmp = alpha * matrix_a * matrix_b + beta * *C;
    *C = tmp;
}

void matrix_matmul_with_transpose(matrix_type *C, const matrix_type *A,
                                  const matrix_type *B, bool transA,
                                  bool transB) {
    matrix_dgemm(C, A, B, transA, transB, 1, 0);
}

/*
   This function does a general matrix multiply of A * B, and stores
   the result in C.
*/

void matrix_matmul(matrix_type *C, const matrix_type *A, const matrix_type *B) {
    *C = *A * *B;
}

/*
   Allocates new matrix C = A·B
*/

matrix_type *matrix_alloc_matmul(const matrix_type *A, const matrix_type *B) {
    return new Eigen::MatrixXd(*A * *B);
}
