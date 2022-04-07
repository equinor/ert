#include <vector>

#include <stdio.h>
#include <string.h>
#include <Eigen/Dense>

#include <ert/util/test_util.h>
#include <ert/util/int_vector.h>
#include <ert/util/type_vector_functions.h>

#include <ert/enkf/meas_data.hpp>

void test_measured_to_matrix() {
    std::vector<bool> ens_mask(10, true);
    meas_data_type *meas_data = meas_data_alloc(ens_mask);
    meas_block_type *meas_block = meas_data_add_block(meas_data, "OBS", 10, 10);

    Eigen::MatrixXd S0 = Eigen::MatrixXd::Zero(10, 10);

    /* We create a simple 10 x 10 matrix and measured data where the data is i*j and
   compare these later when measured data is converted to the S-matrix*/
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 10; i++) {
            S0(i, j) = i * j;
            meas_block_iset(meas_block, i, j, i * j);
        }
    }

    auto S = meas_data_makeS(meas_data);
    test_assert_true(S0 == S);

    meas_data_free(meas_data);
}

int main(int argc, char **argv) {
    test_measured_to_matrix();
    exit(0);
}
