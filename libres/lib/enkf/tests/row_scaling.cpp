/*
   Copyright (C) 2020  Equinor ASA, Norway.

   The file 'row_scaling.cpp' is part of ERT - Ensemble based Reservoir Tool.

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
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ert/util/test_util.hpp>

#include <ert/enkf/row_scaling.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/util/rng.hpp>


void test_create() {
  row_scaling_type * row_scaling = row_scaling_alloc();
  test_assert_int_equal(row_scaling_get_size(row_scaling), 0);

  test_assert_throw(row_scaling_iget(row_scaling, -1), std::out_of_range);
  test_assert_throw(row_scaling_iget(row_scaling, 1000), std::out_of_range);

  test_assert_throw(row_scaling_iset(row_scaling, -1, 0), std::out_of_range);
  test_assert_throw(row_scaling_iset(row_scaling, 0, -1), std::invalid_argument);
  test_assert_throw(row_scaling_iset(row_scaling, 0, 2), std::invalid_argument);

  row_scaling_iset(row_scaling, 9, 0.25);
  test_assert_double_equal(row_scaling_iget(row_scaling, 9), 0.25);

  row_scaling_free(row_scaling);
}

static void scaleX(matrix_type* X, const matrix_type * X0, double alpha) {
  matrix_assign(X, X0);
  matrix_scale(X, alpha);
  for (int i=0; i < matrix_get_rows(X); i++)
    matrix_iset(X, i, i, (1 - alpha) + matrix_iget(X,i,i));
}

void row_scaling_multiply2(const row_scaling_type * row_scaling, matrix_type * A, const matrix_type * X0) {
  matrix_type * X = matrix_alloc(matrix_get_rows(X0), matrix_get_columns(X0));
  for (int row=0; row < row_scaling_get_size(row_scaling); row++) {
    double alpha = row_scaling_iget(row_scaling, row);
    scaleX(X, X0, alpha);

    std::vector<double> row_data(matrix_get_columns(A));
    for (int j=0; j < matrix_get_columns(A); j++) {
      double sum = 0;
      for (int i=0; i < matrix_get_columns(A); i++)
        sum += matrix_iget(A, row, i) * matrix_iget(X,i,j);

      row_data[j] = sum;
    }
    matrix_set_row(A, row_data.data(), row);
  }
  matrix_free(X);
}

void test_multiply(const row_scaling_type * row_scaling, const matrix_type * A0, const matrix_type * X0) {
  matrix_type * A1 = matrix_alloc_copy(A0);
  matrix_type * A2 = matrix_alloc_copy(A0);

  row_scaling_multiply(row_scaling, A1, X0);
  row_scaling_multiply2(row_scaling, A2, X0);
  test_assert_true(matrix_equal(A1, A2));

  matrix_free(A1);
  matrix_free(A2);
}


void test_multiply() {
  const int data_size = 200000;
  const int ens_size  = 100;
  matrix_type * A0 = matrix_alloc(data_size, ens_size);
  matrix_type * X0 = matrix_alloc(ens_size, ens_size);
  rng_type * rng = rng_alloc(MZRAN, INIT_DEFAULT);
  matrix_random_init(A0, rng);

  const int project_iens = 4;
  for (int col=0; col < ens_size; col++)
    matrix_iset(X0, project_iens, col, 1.0);


  // alpha == 1: Full update, should project out realizations project_iens
  {
    row_scaling_type * row_scaling = row_scaling_alloc();
    matrix_type * A = matrix_alloc_copy(A0);
    for (int row = 0; row < data_size; row++)
      row_scaling_iset(row_scaling, row, 1);
    row_scaling_multiply(row_scaling, A, X0);

    for (int row = 0; row < data_size; row++)
      for (int col=0; col < ens_size; col++)
        test_assert_double_equal(matrix_iget(A, row, col), matrix_iget(A0, row, project_iens));

    matrix_free(A);
    test_multiply(row_scaling, A0, X0);
    row_scaling_free(row_scaling);
  }

  // alpha == 0: No update - should have A == A0
  {
    row_scaling_type * row_scaling = row_scaling_alloc();
    matrix_type * A = matrix_alloc_copy(A0);
    std::vector<float> row_data(data_size);
    for (int row = 0; row < data_size; row++)
      row_data[row] = 0;

    row_scaling_assign_float(row_scaling, row_data.data(), row_data.size());
    row_scaling_multiply(row_scaling, A, X0);

    for (int row = 0; row < data_size; row++)
      for (int col=0; col < ens_size; col++)
        test_assert_double_equal(matrix_iget(A, row, col), matrix_iget(A0, row, col));

    matrix_free(A);
    test_multiply(row_scaling, A0, X0);
    row_scaling_free(row_scaling);
  }


  // General alpha
  {
    row_scaling_type * row_scaling = row_scaling_alloc();
    matrix_type * A = matrix_alloc_copy(A0);
    std::vector<double> row_data(data_size);

    row_scaling_iset(row_scaling, 2*data_size, 1.0);
    test_assert_int_equal( row_scaling_get_size(row_scaling), 2*data_size + 1);

    for (int row = 0; row < data_size; row++)
      row_data[row] = rng_get_double(rng);

    row_scaling_assign_double(row_scaling, row_data.data(), row_data.size());
    test_assert_int_equal( row_scaling_get_size(row_scaling), data_size);

    row_scaling_multiply(row_scaling, A, X0);
    for (int row = 0; row < data_size; row++) {
      double alpha = row_scaling_iget(row_scaling, row);
      for (int col=0; col < ens_size; col++) {
        double expected = alpha * matrix_iget(A0, row, project_iens) + (1 - alpha) * matrix_iget(A0, row, col);
        test_assert_double_equal(matrix_iget(A, row, col), expected );
      }
    }

    test_multiply(row_scaling, A0, X0);
    matrix_free(A);
    row_scaling_free(row_scaling);
  }

  rng_free(rng);
  matrix_free(X0);
  matrix_free(A0);
}


int main(int argc , char ** argv) {
  test_create();
  test_multiply();
}

