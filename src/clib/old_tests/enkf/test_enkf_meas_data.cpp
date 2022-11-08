#include <stdlib.h>
#include <vector>

#include <ert/util/int_vector.h>
#include <ert/util/test_util.h>
#include <ert/util/type_vector_functions.h>

#include <ert/enkf/meas_data.hpp>

void create_test() {
    std::vector<size_t> realizations{10, 20, 30};
    {
        meas_data_type *meas_data = meas_data_alloc(realizations);

        {
            meas_block_type *block =
                meas_data_add_block(meas_data, "OBS", 10, 10);

            meas_block_iset(block, 10, 0, 100);
            test_assert_double_equal(100, meas_block_iget(block, 10, 0));
        }
        meas_data_free(meas_data);
    }
}

int main(int argc, char **argv) {
    create_test();
    exit(0);
}
