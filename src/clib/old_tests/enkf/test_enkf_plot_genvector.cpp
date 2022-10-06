#include <stdlib.h>

#include <ert/util/test_util.h>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_plot_genvector.hpp>
#include <ert/enkf/gen_data.hpp>

void test_create() {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc_GEN_DATA_result("Key", ASCII, "Result%d");
    enkf_plot_genvector_type *gen_vector =
        enkf_plot_genvector_alloc(config_node, 0);
    test_assert_true(enkf_plot_genvector_is_instance(gen_vector));
    test_assert_int_equal(0, enkf_plot_genvector_get_size(gen_vector));
    enkf_config_node_free(config_node);
    enkf_plot_genvector_free(gen_vector);
}

int main(int argc, char **argv) {
    test_create();
    exit(0);
}
