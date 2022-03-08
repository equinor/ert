/*
   Copyright (C) 2015  Equinor ASA, Norway.

   The file 'ert_util_matrix_lapack.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <stdlib.h>

#include <ert/util/bool_vector.hpp>
#include <ert/util/test_util.hpp>
#include <ert/res_util/matrix.hpp>

void test_matrix_similar() {
    rng_type *rng = rng_alloc(MZRAN, INIT_DEV_URANDOM);
    matrix_type *m1 = matrix_alloc(3, 3);
    matrix_type *m2 = matrix_alloc(3, 3);
    double epsilon = 0.0000000001;

    matrix_random_init(m1, rng);
    matrix_assign(m2, m1);

    test_assert_true(matrix_similar(m1, m2, epsilon));

    matrix_iadd(m2, 2, 2, 0.5 * epsilon);
    test_assert_true(matrix_similar(m1, m2, epsilon));

    matrix_iadd(m2, 2, 2, epsilon);
    test_assert_true(!matrix_similar(m1, m2, epsilon));

    matrix_free(m2);
    matrix_free(m1);
    rng_free(rng);
}

int main(int argc, char **argv) {
    test_matrix_similar();
    exit(0);
}
