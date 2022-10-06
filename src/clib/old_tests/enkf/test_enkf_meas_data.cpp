#include <stdlib.h>
#include <vector>

#include <ert/util/int_vector.h>
#include <ert/util/test_util.h>
#include <ert/util/type_vector_functions.h>

#include <ert/enkf/meas_data.hpp>

void meas_block_iset_abort(void *arg) {
    meas_block_type *block = meas_block_safe_cast(arg);
    meas_block_iset(block, 0, 0, 100);
}

void meas_block_iget_abort(void *arg) {
    meas_block_type *block = meas_block_safe_cast(arg);
    meas_block_iget(block, 0, 0);
}

void create_test() {
    std::vector<bool> ens_mask(31, false);
    ens_mask[10] = true;
    ens_mask[20] = true;
    ens_mask[30] = true;
    {
        meas_data_type *meas_data = meas_data_alloc(ens_mask);
        test_assert_int_equal(3, meas_data_get_active_ens_size(meas_data));

        {
            meas_block_type *block =
                meas_data_add_block(meas_data, "OBS", 10, 10);

            meas_block_iset(block, 10, 0, 100);
            test_assert_double_equal(100, meas_block_iget(block, 10, 0));

            test_assert_util_abort("meas_block_assert_iens_active",
                                   meas_block_iset_abort, block);
            test_assert_util_abort("meas_block_assert_iens_active",
                                   meas_block_iget_abort, block);
        }
        meas_data_free(meas_data);
    }
}

int main(int argc, char **argv) {
    create_test();
    exit(0);
}
