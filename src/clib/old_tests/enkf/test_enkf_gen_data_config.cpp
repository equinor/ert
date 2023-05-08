#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <ert/enkf/gen_data_config.hpp>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>

void test_report_steps() {
    gen_data_config_type *config = gen_data_config_alloc_GEN_DATA_result();
    test_assert_int_equal(0, gen_data_config_num_report_step(config));
    test_assert_false(gen_data_config_has_report_step(config, 0));

    gen_data_config_add_report_step(config, 10);
    test_assert_int_equal(1, gen_data_config_num_report_step(config));
    test_assert_true(gen_data_config_has_report_step(config, 10));
    test_assert_int_equal(gen_data_config_iget_report_step(config, 0), 10);

    gen_data_config_add_report_step(config, 10);
    test_assert_int_equal(1, gen_data_config_num_report_step(config));
    test_assert_true(gen_data_config_has_report_step(config, 10));

    gen_data_config_add_report_step(config, 5);
    test_assert_int_equal(2, gen_data_config_num_report_step(config));
    test_assert_true(gen_data_config_has_report_step(config, 10));
    test_assert_int_equal(gen_data_config_iget_report_step(config, 0), 5);
    test_assert_int_equal(gen_data_config_iget_report_step(config, 1), 10);

    gen_data_config_free(config);
}

int main(int argc, char **argv) {
    test_report_steps();
    exit(0);
}
