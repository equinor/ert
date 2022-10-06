#include <stdlib.h>

#include <ert/util/test_util.h>

#include <ert/enkf/ensemble_config.hpp>

void add_NULL_node(void *arg) {
    ensemble_config_type *ens_config = ensemble_config_safe_cast(arg);
    ensemble_config_add_node(ens_config, NULL);
}

void test_abort_on_add_NULL() {
    ensemble_config_type *ensemble_config =
        ensemble_config_alloc(NULL, NULL, NULL);

    test_assert_true(ensemble_config_is_instance(ensemble_config));
    test_assert_util_abort("ensemble_config_add_node", add_NULL_node,
                           ensemble_config);

    ensemble_config_free(ensemble_config);
}

int main(int argc, char **argv) {
    test_abort_on_add_NULL();
    exit(0);
}
