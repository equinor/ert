#include <algorithm>

#include <string>

#include <stdlib.h>
#include <stdio.h>

#include <stdexcept>
#include <cmath>

#include <fmt/format.h>
#include <ert/util/ert_api_config.hpp>
#include <ert/res_util/thread_pool.hpp>
#include <ert/util/util.hpp>
#include <ert/util/rng.hpp>

#include <ert/res_util/arg_pack.hpp>
#include <ert/res_util/matrix.hpp>

matrix_type *matrix_alloc(int rows, int columns) {
    if (rows <= 0 || columns <= 0)
        return nullptr;
    auto matrix = new Eigen::MatrixXd;
    matrix->setZero(rows, columns);
    return matrix;
}

matrix_type *matrix_alloc_identity(int dim) {
    auto matrix = new Eigen::MatrixXd;
    matrix->setIdentity(dim, dim);
    return matrix;
}

matrix_type *matrix_alloc_copy(const matrix_type *src) {
    return new Eigen::MatrixXd(*src);
}

void matrix_copy_block(matrix_type *dst_matrix, int dst_row, int dst_column,
                       int rows, int columns, const matrix_type *src_matrix,
                       int src_row, int src_column) {
    dst_matrix->block(dst_row, dst_column, rows, columns) =
        src_matrix->block(src_row, src_column, rows, columns);
}

bool matrix_resize(matrix_type *matrix, int rows, int columns,
                   bool copy_content) {
    matrix->conservativeResize(rows, columns);
    return true;
}

void matrix_free(matrix_type *matrix) { delete matrix; }

/*
     [ a11   a12  ]
     [ a21   a22  ]




   row_major_order == true
   -----------------------
   a_11
   a_12
   a_21
   a_22


   row_major_order == false
   -----------------------
   a_11
   a_12
   a_21
   a_22


   The @orw_major_order parameter ONLY affects the layout on the file,
   and NOT the memory layout of the matrix.
*/

static void __fscanf_and_set(matrix_type *matrix, int row, int col,
                             FILE *stream) {
    double value;
    if (fscanf(stream, "%lg", &value) == 1)
        matrix_iset(matrix, row, col, value);
    else
        throw std::runtime_error(fmt::format(
            "reading of matrix failed at row:{}  col:{}", row, col));
}

void matrix_fscanf_data(matrix_type *matrix, bool row_major_order,
                        FILE *stream) {
    int row, col;
    if (row_major_order) {
        for (row = 0; row < matrix->rows(); row++) {
            for (col = 0; col < matrix->cols(); col++) {
                __fscanf_and_set(matrix, row, col, stream);
            }
        }
    } else {
        for (col = 0; col < matrix->cols(); col++) {
            for (row = 0; row < matrix->rows(); row++) {
                __fscanf_and_set(matrix, row, col, stream);
            }
        }
    }
}

/*
  If the matrix is printed in row_major_order it is printed so that it looks
  visually like a matrix; i.e. with one row on each line in the file. When it is
  written with row_major_order == false it is just written with one number on
  each line.

  As long as the matrix is loaded again with the matrix_fscanf_data() - or a
  similar function which is not line oriented, the presence of newlines make no
  difference anyway.
*/

void matrix_fprintf_data(const matrix_type *matrix, bool row_major_order,
                         FILE *stream) {
    int i, j;
    if (row_major_order) {
        for (i = 0; i < matrix->rows(); i++) {
            for (j = 0; j < matrix->cols(); j++)
                fprintf(stream, "%lg ", matrix_iget(matrix, i, j));
            fprintf(stream, "\n");
        }
    } else {
        for (j = 0; j < matrix->cols(); j++)
            for (i = 0; i < matrix->rows(); i++) {
                fprintf(stream, "%lg\n", matrix_iget(matrix, i, j));
            }
    }
}

static void matrix_assert_ij(const matrix_type *matrix, int i, int j) {
    if ((i < 0) || (i >= matrix->rows()) || (j < 0) || (j >= matrix->cols()))
        throw std::runtime_error(
            fmt::format("(i,j) = ({},{}) invalid. Matrix size: {} x {}", i, j,
                        matrix->rows(), matrix->cols()));
}

static void matrix_assert_equal_rows(const matrix_type *m1,
                                     const matrix_type *m2) {
    if (m1->rows() != m2->rows())
        throw std::runtime_error(
            fmt::format("size mismatch in binary matrix operation {} {}",
                        m1->rows(), m2->rows()));
}

static void matrix_assert_equal_columns(const matrix_type *m1,
                                        const matrix_type *m2) {
    if (m1->cols() != m2->cols())
        throw std::runtime_error(
            fmt::format("size mismatch in binary matrix operation {} {}",
                        m1->cols(), m2->cols()));
}

extern "C" void matrix_set_all(matrix_type *matrix, double value) {
    matrix->setConstant(value);
}

void matrix_iset(matrix_type *matrix, int i, int j, double value) {
    (*matrix)(i, j) = value;
}

void matrix_iset_safe(matrix_type *matrix, int i, int j, double value) {
    matrix_assert_ij(matrix, i, j);
    matrix_iset(matrix, i, j, value);
}

double matrix_iget(const matrix_type *matrix, int i, int j) {
    return (*matrix)(i, j);
}

void matrix_iadd(matrix_type *matrix, int i, int j, double value) {
    (*matrix)(i, j) += value;
}

void matrix_isub(matrix_type *matrix, int i, int j, double value) {
    (*matrix)(i, j) -= value;
}

void matrix_imul(matrix_type *matrix, int i, int j, double value) {
    (*matrix)(i, j) *= value;
}

void matrix_set(matrix_type *matrix, double value) {
    (*matrix).setConstant(value);
}

void matrix_shift(matrix_type *matrix, double value) {
    int i, j;
    for (j = 0; j < matrix->cols(); j++)
        for (i = 0; i < matrix->rows(); i++)
            matrix_iadd(matrix, i, j, value);
}

void matrix_scale(matrix_type *matrix, double value) {
    int i, j;
    for (j = 0; j < matrix->cols(); j++)
        for (i = 0; i < matrix->rows(); i++)
            matrix_imul(matrix, i, j, value);
}

void matrix_scale_row(matrix_type *matrix, int row, double factor) {
    matrix->row(row) *= factor;
}

void matrix_scale_column(matrix_type *matrix, int column, double factor) {
    matrix->col(column) *= factor;
}

void matrix_set_many_on_column(matrix_type *matrix, int row_offset,
                               int elements, const double *data, int column) {
    if ((row_offset + elements) <= matrix->rows()) {
        int i;
        for (i = 0; i < elements; i++)
            (*matrix)(row_offset + i, column) = data[i];
    } else
        throw std::out_of_range("range violation");
}

void matrix_copy_column(matrix_type *dst_matrix, const matrix_type *src_matrix,
                        int dst_column, int src_column) {
    matrix_assert_equal_rows(dst_matrix, src_matrix);
    {
        int row;
        for (row = 0; row < dst_matrix->rows(); row++)
            (*dst_matrix)(row, dst_column) = (*src_matrix)(row, src_column);
    }
}

double matrix_column_column_dot_product(const matrix_type *m1, int col1,
                                        const matrix_type *m2, int col2) {
    return m1->col(col1).dot(m2->col(col2));
}

void matrix_assign(matrix_type *A, const matrix_type *B) { *A = *B; }

void matrix_inplace_add(matrix_type *A, const matrix_type *B) { *A += *B; }

void matrix_inplace_sub(matrix_type *A, const matrix_type *B) { *A -= *B; }

void matrix_sub(matrix_type *A, const matrix_type *B, const matrix_type *C) {
    *A = *B - *C;
}

void matrix_transpose(const matrix_type *A, matrix_type *T) {
    *T = A->transpose();
}

void matrix_inplace_transpose(matrix_type *A) { A->transposeInPlace(); }

matrix_type *matrix_alloc_transpose(const matrix_type *A) {
    return new Eigen::MatrixXd(A->transpose());
}

void matrix_inplace_matmul(matrix_type *A, const matrix_type *B) { *A *= *B; }

double matrix_get_row_sum(const matrix_type *matrix, int row) {
    return matrix->row(row).sum();
}

void matrix_shift_row(matrix_type *matrix, int row, double shift) {
    int j;
    for (j = 0; j < matrix->cols(); j++)
        (*matrix)(row, j) += shift;
}

void matrix_set_row(matrix_type *matrix, const double *data, int row) {
    if (row < 0 || row >= matrix->rows())
        throw std::invalid_argument("Invalid row index");

    for (int j = 0; j < matrix->cols(); j++)
        (*matrix)(row, j) = data[j];
}

/*
   For each row in the matrix we will do the operation

     R -> R - <R>
*/

/**
 * Subtract mean from each row of matrix.
 * 
 * In literature typically defined using the following matrix computations:
 * 
 * (I_N - 1/N \bm{1} \times \bm{1}^T)
 * 
 * where I_N is the N-dimensional identity matrix and
 * \bm{1} is an N-vector with all elements equal to 1.
 * See for example (Eq. 12) in the paper 
 * Efficient Implementation of an Iterative Ensemble Smoother for 
 * Data Assimilation and Reservoir History Matching, 2019, Evensen.
 * 
 * Also see tests for more details.
 */
void matrix_subtract_row_mean(matrix_type *matrix) {
    int i;
    for (i = 0; i < matrix->rows(); i++) {
        double row_mean = matrix_get_row_sum(matrix, i) / matrix->cols();
        matrix_shift_row(matrix, i, -row_mean);
    }
}

int matrix_get_rows(const matrix_type *matrix) { return matrix->rows(); }

int matrix_get_columns(const matrix_type *matrix) { return matrix->cols(); }

bool matrix_equal(const matrix_type *m1, const matrix_type *m2) {
    if (m1->rows() != m2->rows() || m1->cols() != m2->cols())
        return false;
    return m1->isApprox(*m2);
}

/*
   Returns true if the two matrices m1 and m2 are almost equal.
   The equality-test applies an element-by-element comparison
   whether the absolute value of the difference is less than a
   user-specified tolerance. This routine is useful for testing
   two different numerical algorithms that should give the same
   result but not necessarily to machine precision. If the two
   matrices do not have equal dimension false is returned.
*/

bool matrix_similar(const matrix_type *m1, const matrix_type *m2,
                    double epsilon) {
    if (!((m1->rows() == m2->rows()) && (m1->cols() == m2->cols())))
        return false;
    {
        for (int i = 0; i < m1->rows(); i++) {
            for (int j = 0; j < m1->cols(); j++) {
                double d1 = (*m1)(i, j);
                double d2 = (*m2)(i, j);

                if (std::abs(d1 - d2) > epsilon)
                    return false;
            }
        }
    }

    /* OK - we came all the way through - they are almost equal. */
    return true;
}

bool matrix_columns_equal(const matrix_type *m1, int col1,
                          const matrix_type *m2, int col2) {
    return m1->col(col1).isApprox(m2->col(col2));
}

void matrix_diag_set_scalar(matrix_type *matrix, double value) {
    if (matrix->rows() == matrix->cols()) {
        int i;
        matrix_set(matrix, 0);
        for (i = 0; i < matrix->rows(); i++)
            (*matrix)(i, i) = value;
    } else
        std::runtime_error("size mismatch");
}

void matrix_random_init(matrix_type *matrix, rng_type *rng) {
    int i, j;
    for (j = 0; j < matrix->cols(); j++)
        for (i = 0; i < matrix->rows(); i++)
            (*matrix)(i, j) = rng_get_double(rng);
}

void matrix_delete_column(matrix_type *m1, int column) {
    if (column < 0 || column >= matrix_get_columns(m1))
        throw std::invalid_argument("Invalid column" + std::to_string(column));

    matrix_type *m2 =
        matrix_alloc(matrix_get_rows(m1), matrix_get_columns(m1) - 1);
    if (column > 0)
        matrix_copy_block(m2, 0, 0, matrix_get_rows(m2), column, m1, 0, 0);

    if (column < (matrix_get_columns(m1) - 1))
        matrix_copy_block(m2, 0, column, matrix_get_rows(m2),
                          matrix_get_columns(m2) - column, m1, 0, column + 1);

    matrix_resize(m1, matrix_get_rows(m2), matrix_get_columns(m2), false);
    matrix_assign(m1, m2);
    matrix_free(m2);
}

void matrix_delete_row(matrix_type *m1, int row) {
    if (row < 0 || row >= matrix_get_rows(m1))
        throw std::invalid_argument("Invalid row" + std::to_string(row));

    matrix_type *m2 =
        matrix_alloc(matrix_get_rows(m1) - 1, matrix_get_columns(m1));
    if (row > 0)
        matrix_copy_block(m2, 0, 0, row, matrix_get_columns(m2), m1, 0, 0);

    if (row < (matrix_get_rows(m1) - 1))
        matrix_copy_block(m2, row, 0, matrix_get_rows(m2) - row,
                          matrix_get_columns(m2), m1, row + 1, 0);

    matrix_resize(m1, matrix_get_rows(m2), matrix_get_columns(m2), false);
    matrix_assign(m1, m2);
    matrix_free(m2);
}

bool matrix_check_dims(const matrix_type *m, int rows, int columns) {
    if (m) {
        if ((m->rows() == rows) && (m->cols() == columns))
            return true;
        else
            return false;
    } else {
        throw std::runtime_error("trying to dereference NULL matrix pointer");
    }
}
