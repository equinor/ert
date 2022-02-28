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

/*
   This little function translates between an integer identifier
   (i.e. and enum instance) to one of the characters used by the low
   level lapack routine to indicate how the singular vectors should be
   returned to the calling scope.

   The meaning of the different enum values is documented in the enum
   definition in the header file matrix_lapack.h.
*/

static char dgesvd_get_vector_job(dgesvd_vector_enum vector_job) {
    char job = 'X';
    switch (vector_job) {
    case (DGESVD_ALL):
        job = 'A';
        break;
    case (DGESVD_MIN_RETURN):
        job = 'S';
        break;
    case (DGESVD_MIN_OVERWRITE):
        job = 'O';
        break;
    case (DGESVD_NONE):
        job = 'N';
        break;
    default:
        util_abort("%s: internal error - unrecognized code:%d \n", vector_job);
    }
    return job;
}

/*
   If jobu == DGSEVD_NONE the U matrix can be NULL, same for jobvt.
*/

void matrix_dgesvd(dgesvd_vector_enum jobu, dgesvd_vector_enum jobvt,
                   matrix_type *A, double *S, matrix_type *U, matrix_type *VT) {

    int opts = 0;
    if (U != nullptr)
        opts |= jobu == DGESVD_ALL ? Eigen::ComputeFullU : Eigen::ComputeThinU;
    if (VT != nullptr)
        opts |= jobvt == DGESVD_ALL ? Eigen::ComputeFullV : Eigen::ComputeThinV;

    auto svd = A->bdcSvd(opts);

    if (U != nullptr)
        *U = svd.matrixU();
    if (VT != nullptr)
        *VT = svd.matrixV().transpose();

    auto singular = svd.singularValues();
    std::copy(singular.begin(), singular.end(), S);
}
