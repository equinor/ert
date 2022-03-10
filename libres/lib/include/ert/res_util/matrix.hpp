/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'matrix.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_MATRIX_H
#define ERT_MATRIX_H
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <Eigen/Dense>

#include <ert/util/ert_api_config.hpp>
#include <ert/util/rng.hpp>
#include <ert/util/type_macros.hpp>
#include <ert/util/bool_vector.hpp>

using matrix_type = Eigen::MatrixXd;

bool matrix_check_dims(const matrix_type *m, int rows, int columns);
void matrix_fscanf_data(matrix_type *matrix, bool row_major_order,
                        FILE *stream);
void matrix_fprintf_data(const matrix_type *matrix, bool row_major_order,
                         FILE *stream);
void matrix_fprintf(const matrix_type *matrix, const char *fmt, FILE *stream);
void matrix_dump_csv(const matrix_type *matrix, const char *filename);
extern "C" matrix_type *matrix_alloc(int rows, int columns);
extern "C" matrix_type *matrix_alloc_identity(int dim);
bool matrix_resize(matrix_type *matrix, int rows, int columns,
                   bool copy_content);
matrix_type *matrix_alloc_sub_copy(const matrix_type *src, int row_offset,
                                   int column_offset, int rows, int columns);
extern "C" matrix_type *matrix_alloc_copy(const matrix_type *src);
void matrix_column_compressed_memcpy(matrix_type *target,
                                     const matrix_type *src,
                                     const bool_vector_type *mask);
matrix_type *matrix_realloc_copy(matrix_type *T, const matrix_type *src);
void matrix_set_row(matrix_type *matrix, const double *data, int row);

matrix_type *matrix_alloc_shared(const matrix_type *src, int row, int column,
                                 int rows, int columns);
extern "C" void matrix_free(matrix_type *matrix);
void matrix_set(matrix_type *matrix, double value);
void matrix_scale(matrix_type *matrix, double value);
void matrix_shift(matrix_type *matrix, double value);

void matrix_assign(matrix_type *A, const matrix_type *B);
void matrix_inplace_add(matrix_type *A, const matrix_type *B);
void matrix_inplace_sub(matrix_type *A, const matrix_type *B);
void matrix_sub(matrix_type *A, const matrix_type *B, const matrix_type *C);
void matrix_transpose(const matrix_type *A, matrix_type *T);
void matrix_inplace_add_column(matrix_type *A, const matrix_type *B, int colA,
                               int colB);
extern "C" void matrix_inplace_transpose(matrix_type *A);

void matrix_iset_safe(matrix_type *matrix, int i, int j, double value);
extern "C" void matrix_iset(matrix_type *matrix, int i, int j, double value);
extern "C" double matrix_iget(const matrix_type *matrix, int i, int j);
double matrix_iget_safe(const matrix_type *matrix, int i, int j);
void matrix_iadd(matrix_type *matrix, int i, int j, double value);
void matrix_isub(matrix_type *matrix, int i, int j, double value);
void matrix_imul(matrix_type *matrix, int i, int j, double value);

void matrix_inplace_matmul(matrix_type *A, const matrix_type *B);

void matrix_shift_row(matrix_type *matrix, int row, double shift);
double matrix_get_column_sum(const matrix_type *matrix, int column);
double matrix_get_row_sum(const matrix_type *matrix, int column);
double matrix_get_column_sum2(const matrix_type *matrix, int column);
double matrix_get_column_abssum(const matrix_type *matrix, int column);
void matrix_subtract_row_mean(matrix_type *matrix);
void matrix_subtract_and_store_row_mean(matrix_type *matrix,
                                        matrix_type *row_mean);
extern "C" void matrix_scale_column(matrix_type *matrix, int column,
                                    double scale_factor);
extern "C" void matrix_scale_row(matrix_type *matrix, int row,
                                 double scale_factor);
void matrix_set_const_column(matrix_type *matrix, const double value,
                             int column);

double *matrix_get_data(const matrix_type *matrix);

void matrix_set_column(matrix_type *matrix, const double *data, int column);
void matrix_set_many_on_column(matrix_type *matrix, int row_offset,
                               int elements, const double *data, int column);
void matrix_full_size(matrix_type *matrix);
extern "C" int matrix_get_rows(const matrix_type *matrix);
extern "C" int matrix_get_columns(const matrix_type *matrix);
int matrix_get_column_stride(const matrix_type *matrix);
void matrix_get_dims(const matrix_type *matrix, int *rows, int *columns,
                     int *row_stride, int *column_stride);
bool matrix_is_quadratic(const matrix_type *matrix);
extern "C" bool matrix_equal(const matrix_type *m1, const matrix_type *m2);

void matrix_diag_set_scalar(matrix_type *matrix, double value);
void matrix_diag_set(matrix_type *matrix, const double *diag);
extern "C" void matrix_random_init(matrix_type *matrix, rng_type *rng);
void matrix_delete_row(matrix_type *m1, int row);
void matrix_delete_column(matrix_type *m1, int row);

void matrix_imul_col(matrix_type *matrix, int column, double factor);
double matrix_column_column_dot_product(const matrix_type *m1, int col1,
                                        const matrix_type *m2, int col2);
double matrix_row_column_dot_product(const matrix_type *m1, int row1,
                                     const matrix_type *m2, int col2);
extern "C" matrix_type *matrix_alloc_transpose(const matrix_type *A);
void matrix_copy_row(matrix_type *target_matrix, const matrix_type *src_matrix,
                     int target_row, int src_row);
void matrix_copy_block(matrix_type *target_matrix, int target_row,
                       int target_column, int rows, int columns,
                       const matrix_type *src_matrix, int src_row,
                       int src_column);

void matrix_scalar_set(matrix_type *matrix, double value);
void matrix_inplace_diag_sqrt(matrix_type *Cd);
double matrix_trace(const matrix_type *matrix);

#define matrix_shrink_header(_Mat, _Row, _Col)                                 \
    matrix_resize((_Mat), (_Row), (_Col), true)
#endif
