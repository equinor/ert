#include <stdlib.h>

#include <ert/util/test_util.h>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_plot_gen_kw.hpp>

void test_create_invalid() {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc_summary("WWCT", LOAD_FAIL_SILENT);
    enkf_plot_gen_kw_type *gen_kw = enkf_plot_gen_kw_alloc(config_node);

    test_assert_NULL(gen_kw);
    enkf_config_node_free(config_node);
}

void test_create() {
    enkf_config_node_type *config_node =
        enkf_config_node_new_gen_kw("GEN_KW", DEFAULT_GEN_KW_TAG_FORMAT, false);

    {
        enkf_plot_gen_kw_type *gen_kw = enkf_plot_gen_kw_alloc(config_node);
        test_assert_true(enkf_plot_gen_kw_is_instance(gen_kw));
        enkf_plot_gen_kw_free(gen_kw);
    }

    enkf_config_node_free(config_node);
}

int main(int argc, char **argv) {
    test_create();
    test_create_invalid();
    exit(0);
}
