#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <ert/enkf/gen_data_config.hpp>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>

void test_report_steps() {
    gen_data_config_type *config =
        gen_data_config_alloc_GEN_DATA_result("KEY", ASCII);
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

void alloc_invalid_io_format1(void *arg) {
    gen_data_config_type *config =
        gen_data_config_alloc_GEN_DATA_result("KEY", ASCII_TEMPLATE);
    gen_data_config_free(config);
}

void test_set_invalid_format() {
    test_assert_util_abort("gen_data_config_alloc_GEN_DATA_result",
                           alloc_invalid_io_format1, NULL);
}

int main(int argc, char **argv) {

    const char *gendata_file = argv[1];
    const char *gendata_file_empty = argv[2];
    util_install_signals();
    test_report_steps();
    test_set_invalid_format();

    exit(0);
}
