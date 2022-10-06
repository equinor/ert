#include <stdlib.h>

#include <ert/util/test_util.h>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_plot_gen_kw_vector.hpp>

void test_create() {
    enkf_config_node_type *config_node =
        enkf_config_node_new_gen_kw("GEN_KW", DEFAULT_GEN_KW_TAG_FORMAT, false);

    enkf_plot_gen_kw_vector_type *vector =
        enkf_plot_gen_kw_vector_alloc(config_node, 0);
    test_assert_true(enkf_plot_gen_kw_vector_is_instance(vector));
    test_assert_int_equal(0, enkf_plot_gen_kw_vector_get_size(vector));

    enkf_plot_gen_kw_vector_free(vector);
    enkf_config_node_free(config_node);
}

int main(int argc, char **argv) {
    test_create();
    exit(0);
}
