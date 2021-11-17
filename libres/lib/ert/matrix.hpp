#pragma once

#include <algorithm>
#include <initializer_list>
#include <memory>

#include <ert/res_util/matrix.hpp>

extern "C" matrix_type *matrix_alloc_matmul(const matrix_type *A,
                                            const matrix_type *B);

namespace ert::detail {
struct matrix_deleter {
    void operator()(matrix_type *matrix) const { matrix_free(matrix); }
};
} // namespace ert::detail

namespace ert {

class MatrixXd {
    std::unique_ptr<matrix_type, detail::matrix_deleter> m_matrix;

    MatrixXd(matrix_type *matrix) : m_matrix(matrix) {}

public:
    MatrixXd() {}
    MatrixXd(int rows, int cols) : m_matrix(matrix_alloc(rows, cols)) {}
    MatrixXd(const MatrixXd &other)
        : m_matrix(matrix_alloc_copy(other.get())) {}
    MatrixXd(const std::initializer_list<std::initializer_list<double>> &il);

    matrix_type *get() { return m_matrix.get(); }
    const matrix_type *get() const { return m_matrix.get(); }

    MatrixXd &operator=(const MatrixXd &other) {
        m_matrix.reset(matrix_alloc_copy(other.get()));
        return *this;
    }

    MatrixXd operator+(const MatrixXd &other) const {
        auto self = *this;
        self += other;
        return self;
    }

    MatrixXd &operator+=(const MatrixXd &other) {
        matrix_inplace_add(get(), other.get());
        return *this;
    }

    MatrixXd operator-(const MatrixXd &other) const {
        auto self = *this;
        self -= other;
        return self;
    }

    MatrixXd &operator-=(const MatrixXd &other) {
        matrix_inplace_sub(get(), other.get());
        return *this;
    }

    MatrixXd operator*(const MatrixXd &other) const {
        return {matrix_alloc_matmul(get(), other.get())};
    }

    MatrixXd &operator*=(const MatrixXd &other) {
        auto self = *this * other;
        *this = self;
        return *this;
    }

    int rows() const { return m_matrix ? m_matrix->rows : 0; }
    int cols() const { return m_matrix ? m_matrix->columns : 0; }

    double operator()(int row, int col) const {
        return matrix_iget_safe(get(), row, col);
    }

    MatrixXd transpose() const { return {matrix_alloc_transpose(get())}; }
};
} // namespace ert

inline ert::MatrixXd::MatrixXd(
    const std::initializer_list<std::initializer_list<double>> &il) {
    int rows = il.size();
    int cols = 0;
    for (auto &row : il)
        cols = std::max(cols, static_cast<int>(row.size()));

    m_matrix.reset(matrix_alloc(rows, cols));

    int row = 0;
    for (auto &row_il : il) {
        int col = 0;
        for (auto cell : row_il) {
            matrix_iset(get(), row, col, cell);
            ++col;
        }
        ++row;
    }
}
