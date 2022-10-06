#include <stdlib.h>

#include <ert/util/test_util.h>

#include <ert/enkf/enkf_plot_gendata.hpp>

void test_create_invalid() {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc_summary("WWCT", LOAD_FAIL_SILENT);
    obs_vector_type *obs_vector =
        obs_vector_alloc(SUMMARY_OBS, "OBS", config_node, 100);
    enkf_plot_gendata_type *gen_data =
        enkf_plot_gendata_alloc_from_obs_vector(obs_vector);
    test_assert_NULL(gen_data);
    enkf_config_node_free(config_node);
    obs_vector_free(obs_vector);
}

void test_create() {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc_GEN_DATA_result("key", ASCII, "Result:%d");
    enkf_plot_gendata_type *gen_data = enkf_plot_gendata_alloc(config_node);
    test_assert_true(enkf_plot_gendata_is_instance(gen_data));
    test_assert_int_equal(0, enkf_plot_gendata_get_size(gen_data));
    enkf_config_node_free(config_node);
    enkf_plot_gendata_free(gen_data);
}

int main(int argc, char **argv) {
    test_create();
    test_create_invalid();
    exit(0);
}
