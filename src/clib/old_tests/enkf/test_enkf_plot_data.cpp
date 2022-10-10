#include <stdlib.h>

#include <ert/util/test_util.h>

#include <ert/enkf/enkf_plot_data.hpp>
#include <ert/enkf/enkf_plot_tvector.hpp>

void test_create() {
    enkf_plot_data_type *plot_data = enkf_plot_data_alloc(NULL);
    test_assert_true(enkf_plot_data_is_instance(plot_data));
    enkf_plot_data_free(plot_data);
}

int main(int argc, char **argv) { test_create(); }
