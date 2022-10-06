#ifndef ERT_ENKF_SERIALIZE_H
#define ERT_ENKF_SERIALIZE_H

#include <Eigen/Dense>
#include <ert/ecl/ecl_type.h>
#include <ert/enkf/active_list.hpp>

void enkf_matrix_serialize(const void *__node_data, int node_size,
                           ecl_data_type node_type,
                           const ActiveList &active_list, Eigen::MatrixXd &A,
                           int row_offset, int column);

void enkf_matrix_deserialize(void *__node_data, int node_size,
                             ecl_data_type node_type,
                             const ActiveList &active_list,
                             const Eigen::MatrixXd &A, int row_offset,
                             int column);

#endif
