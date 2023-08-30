#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <ert/enkf/row_scaling.hpp>
#include <ert/python.hpp>

void scaleX(Eigen::MatrixXd &X, const Eigen::MatrixXd &X0, double alpha) {
    X = X0;
    X *= alpha;
    X.diagonal().array() += (1 - alpha);
}

size_t RowScaling::size() const { return m_data.size(); }

double RowScaling::operator[](size_t index) const { return m_data.at(index); }

/**
  The values in the row_scaling container are distributed among
  `m_resolution` discrete values. The reason for this lumping is to be
  able to reuse the scaled update matrix:

      X(alpha) = alpha*X0 + (1 - alpha)*I

  for several rows.
*/
double RowScaling::clamp(double value) const {
    return floor(value * this->m_resolution) / this->m_resolution;
}

void RowScaling::m_resize(size_t new_size) {
    const double default_value = 0;
    m_data.resize(new_size, default_value);
}

double RowScaling::assign(size_t index, double value) {
    if (value < 0 || value > 1)
        throw std::invalid_argument("Invalid value");

    if (m_data.size() <= index)
        m_resize(index + 1);

    m_data.at(index) = clamp(value);
    return m_data.at(index);
}

/**
  The final step in the Ensemble Smoother update is the matrix multiplication

     A' = A * X

  The core of the row_scaling approach is that for each row i in the A matrix
  the X matrix transformed as:

      X(alpha) = (1 - alpha)*I + alpha*X

  where 0 <= alpha <= 1 denotes the 'strength' of the update; alpha == 1
  corresponds to a normal smoother update and alpha == 0 corresponds to no
  update. With the per row transformation of X the operation is no longer matrix
  multiplication but the pseudo code looks like:

     A'(i,j) = \sum_{k} A(i,k) * X'(k,j)

  With X' = X(alpha(i)). When alpha varies as a function of the row index 'i'
  the X matrix should be recalculated for every row. To reduce the number of X
  recalculations the row_scaling values are fixed to a finite number of values
  (given by the resolution member in the row_scaling class) and the
  multiplications are grouped together where all rows with the same alpha valued
  are multiplied in one go.
 */
void RowScaling::multiply(Eigen::Ref<Eigen::MatrixXd> A,
                          const Eigen::MatrixXd &X0) const {
    if (m_data.size() != A.rows())
        throw std::invalid_argument(
            "Size mismatch between row_scaling and A matrix");

    if (A.cols() != X0.rows())
        throw std::invalid_argument("Size mismatch between X0 and A matrix");

    if (X0.cols() != X0.rows())
        throw std::invalid_argument("X0 matrix is not quadratic");

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(X0.rows(), X0.cols());

    // The sort_index vector is an index permutation corresponding to sorted
    // row_scaling data.
    std::vector<int> sort_index(this->size());
    std::iota(sort_index.begin(), sort_index.end(), 0);
    std::sort(sort_index.begin(), sort_index.end(),
              [this](int index1, int index2) {
                  return this->operator[](index1) > this->operator[](index2);
              });

    // This is a double while loop where we eventually go through all the rows
    // in the A matrix / row_scaling vector. In the inner loop we identify a
    // list of rows which have the same row_scaling value, then we scale the X
    // matrix according to this shared alpha value and calculate the update.
    std::size_t index_offset = 0;
    while (true) {
        if (index_offset == m_data.size())
            break;

        auto end_index = index_offset;
        auto current_alpha = m_data[sort_index[end_index]];
        std::vector<int> row_list;

        // 1: Identify rows with the same alpha value
        while (true) {
            if (end_index == m_data.size())
                break;

            if (m_data[sort_index[end_index]] != current_alpha)
                break;

            row_list.push_back(sort_index[end_index]);
            end_index += 1;
        }
        if (row_list.empty())
            break;

        double alpha = m_data[row_list[0]];
        if (alpha == 0.0)
            break;

        // 2: Calculate the scaled X matrix
        scaleX(X, X0, alpha);

        // 3: Calculate A' = A * X for the rows with the same alpha
        for (const auto &row : row_list) {
            std::vector<double> target_row(A.cols());
            Eigen::VectorXd src_row = A.row(row);

            for (int j = 0; j < A.cols(); j++) {
                target_row[j] = src_row.cwiseProduct(X.col(j)).sum();
            }

            if (row < 0 || row >= A.rows())
                throw std::invalid_argument("Invalid row index");

            for (int j = 0; j < A.cols(); j++)
                A(row, j) = target_row.data()[j];
        }

        index_offset = end_index;
    }
}

void RowScaling::assign_vector(const float *data, size_t size) {
    m_resize(size);
    for (int index = 0; index < size; index++)
        assign(index, data[index]);
}

void RowScaling::assign_vector(const double *data, size_t size) {
    m_resize(size);
    for (int index = 0; index < size; index++)
        assign(index, data[index]);
}

namespace {
void setitem(RowScaling &self, size_t index, double value) {
    self.assign(index, value);
}

double getitem(const RowScaling &self, size_t index) { return self[index]; }

const auto assign_vector_doc = R"(
Assign tapering value for all elements via a vector.

The assign_vector() function will resize the row_scaling vector to the
number of elements in the scaling_vector, and assign a value to all
elements.

A typical situation might be to load the scaling data as a EclKW
instance from a grdecl formatted file. Before the assign_vector() can
be called we must transform to an only active elements representation
and use a numpy view:

    # Load scaling vector from grdecl file; typically created with
    # geomodelling software.
    with open("scaling.grdecl") as fh:
        kw_global = EclKW.read_grdecl(fh, "SCALING")

    # Create a ecl_kw copy with only the active elements.
    kw_active = grid.compressed_kw_copy(kw_global)

    # Create a numpy view and invoke the assign_vector() function
    row_scaling.assign_vector(kw_active.numpy_view())
)";
template <typename T>
void assign_vector(RowScaling &self, const py::array_t<T> &array) {
    self.assign_vector(array.data(), array.size());
}
} // namespace

ERT_CLIB_SUBMODULE("local.row_scaling", m) {
    using namespace py::literals;

    py::options opts;
    opts.disable_function_signatures();

    py::class_<RowScaling, std::shared_ptr<RowScaling>>(m, "RowScaling")
        .def(py::init<>())
        .def("__len__", &RowScaling::size)
        .def("__setitem__", &setitem, "index"_a, "value"_a)
        .def("__getitem__", &getitem, "index"_a)
        .def("clamp", &RowScaling::clamp, "value"_a)
        .def("assign_vector", &assign_vector<float>, py::doc{assign_vector_doc},
             "scaling_vector"_a)
        .def("assign_vector", &assign_vector<double>, "scaling_vector"_a)
        .def("multiply", &RowScaling::multiply);
}
