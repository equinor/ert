#include <ert/util/test_util.hpp>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/ies/ies_config.hpp>

void test_steplength1() {

    analysis_module_type *std_module =
        analysis_module_alloc(100, ENSEMBLE_SMOOTHER);
    analysis_module_type *ies_module =
        analysis_module_alloc(100, ITERATED_ENSEMBLE_SMOOTHER);

    test_assert_true(
        analysis_module_set_var(std_module, ies::ENKF_TRUNCATION_KEY, "0.95"));
    test_assert_true(
        analysis_module_set_var(ies_module, ies::ENKF_TRUNCATION_KEY, "0.95"));

    analysis_module_free(std_module);
    analysis_module_free(ies_module);
}

void test_load() {
    analysis_module_type *module =
        analysis_module_alloc(100, ITERATED_ENSEMBLE_SMOOTHER);
    test_assert_not_NULL(module);
    analysis_module_free(module);
}

int main(int argc, char **argv) {
    const char *path_testdata = argv[1];

    test_load();
    test_steplength1();
}
