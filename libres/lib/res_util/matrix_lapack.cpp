/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'matrix_lapack.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <algorithm>
#include <cmath>
#include <vector>

#include <ert/util/util.hpp>

#include <ert/res_util/matrix.hpp>
#include <ert/res_util/matrix_lapack.hpp>

#include "fmt/format.h"
#include "fmt/ostream.h"

/*
   Solves the linear equations Ax = B. The solution is stored in B on
   return.
*/
void matrix_dgesv(matrix_type *A, matrix_type *B) {
    Eigen::MatrixXd tmp = A->partialPivLu().solve(*B);
    *B = tmp;
}

void matrix_dgesvx(matrix_type *A, matrix_type *B, double *rcond) {
    Eigen::MatrixXd tmp = A->fullPivLu().solve(*B);

    *B = tmp;
}

/* The matrix will be inverted in-place, the inversion is based on LU
   factorisation in the routine matrix_dgetrf__(  ).

   The return value:

     =0 : Success
     >0 : Singular matrix
     <0 : Invalid input
*/
int matrix_inv(matrix_type *A) {
    if (!A->fullPivLu().isInvertible())
        return 1;
    Eigen::MatrixXd tmp = A->fullPivLu().inverse();
    *A = tmp;
    return 0;
}
