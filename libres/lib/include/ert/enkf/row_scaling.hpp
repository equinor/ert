#pragma once
#include <ert/res_util/matrix.hpp>
#include <vector>
#include <memory>

class RowScaling : public std::enable_shared_from_this<RowScaling> {
    size_t m_resolution = 1000;
    std::vector<double> m_data;

    void m_resize(size_t new_size);

public:
    double operator[](size_t index) const;
    double assign(size_t index, double value);
    double clamp(double value) const;
    void multiply(matrix_type *A, const matrix_type *X0) const;
    size_t size() const;

    void assign_vector(const float *data, size_t size);
    void assign_vector(const double *data, size_t size);
};
