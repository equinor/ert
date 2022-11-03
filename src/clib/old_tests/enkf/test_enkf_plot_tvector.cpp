#include <stdlib.h>
#include <unistd.h>

#include <ert/util/test_util.h>

#include <ert/enkf/enkf_plot_tvector.hpp>

void create_test() {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc_summary("KEY", LOAD_FAIL_SILENT);
    enkf_plot_tvector_type *tvector = enkf_plot_tvector_alloc(config_node, 0);
    enkf_plot_tvector_free(tvector);
}

void test_iset() {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc_summary("KEY", LOAD_FAIL_SILENT);
    enkf_plot_tvector_type *tvector = enkf_plot_tvector_alloc(config_node, 0);
    enkf_plot_tvector_iset(tvector, 10, 0, 100);

    test_assert_int_equal(11, enkf_plot_tvector_size(tvector));
    test_assert_double_equal(100, enkf_plot_tvector_iget_value(tvector, 10));
    {
        for (int i = 0; i < (enkf_plot_tvector_size(tvector) - 1); i++)
            test_assert_false(enkf_plot_tvector_iget_active(tvector, i));

        test_assert_true(enkf_plot_tvector_iget_active(tvector, 10));
    }

    enkf_plot_tvector_free(tvector);
}

void test_iget() {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc_summary("KEY", LOAD_FAIL_SILENT);
    enkf_plot_tvector_type *tvector = enkf_plot_tvector_alloc(config_node, 0);
    enkf_plot_tvector_iset(tvector, 0, 0, 0);
    enkf_plot_tvector_iset(tvector, 1, 100, 10);
    enkf_plot_tvector_iset(tvector, 2, 200, 20);
    enkf_plot_tvector_iset(tvector, 3, 300, 30);
    enkf_plot_tvector_iset(tvector, 4, 400, 40);

    enkf_plot_tvector_iset(tvector, 6, 600, 60);

    test_assert_int_equal(7, enkf_plot_tvector_size(tvector));
    for (int i = 0; i < 7; i++) {
        if (i == 5)
            test_assert_false(enkf_plot_tvector_iget_active(tvector, i));
        else {
            test_assert_true(enkf_plot_tvector_iget_active(tvector, i));
            test_assert_double_equal(i * 10,
                                     enkf_plot_tvector_iget_value(tvector, i));
        }
    }
}

int main(int argc, char **argv) {
    create_test();
    test_iset();
    test_iget();

    exit(0);
}
