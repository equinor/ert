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

    {
        const int_vector_type *active_steps =
            gen_data_config_get_active_report_steps(config);

        test_assert_int_equal(int_vector_iget(active_steps, 0), 5);
        test_assert_int_equal(int_vector_iget(active_steps, 1), 10);
    }

    gen_data_config_free(config);
}

void test_result_format() {
    test_assert_true(gen_data_config_valid_result_format("path/file%d/extra"));
    test_assert_true(gen_data_config_valid_result_format("file%04d"));
    test_assert_false(gen_data_config_valid_result_format("/path/file%04d"));

    test_assert_false(gen_data_config_valid_result_format("/path/file%s"));
    test_assert_false(gen_data_config_valid_result_format("/path/file"));
    test_assert_false(gen_data_config_valid_result_format("/path/file%f"));

    test_assert_false(gen_data_config_valid_result_format(NULL));
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

void test_format_check() {
    test_assert_int_equal(GEN_DATA_UNDEFINED,
                          gen_data_config_check_format(NULL));
    test_assert_int_equal(GEN_DATA_UNDEFINED,
                          gen_data_config_check_format("Error?"));
    test_assert_int_equal(ASCII, gen_data_config_check_format("ASCII"));
    test_assert_int_equal(ASCII_TEMPLATE,
                          gen_data_config_check_format("ASCII_TEMPLATE"));
}

int main(int argc, char **argv) {

    const char *gendata_file = argv[1];
    const char *gendata_file_empty = argv[2];
    util_install_signals();
    test_report_steps();
    test_result_format();
    test_set_invalid_format();
    test_format_check();

    exit(0);
}
