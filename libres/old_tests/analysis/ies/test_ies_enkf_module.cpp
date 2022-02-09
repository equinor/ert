#include <ert/util/test_util.hpp>
#include <ert/util/rng.h>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/ies/ies_config.hpp>
#include <ert/res_util/es_testdata.hpp>

void test_steplength1(const char *path_testdata) {
    res::es_testdata testdata(path_testdata);
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    matrix_type *X =
        matrix_alloc(testdata.active_ens_size, testdata.active_ens_size);
    matrix_type *prior = testdata.alloc_matrix("A0", testdata.state_size,
                                               testdata.active_ens_size);

    analysis_module_type *std_module =
        analysis_module_alloc(100, ENSEMBLE_SMOOTHER);
    analysis_module_type *ies_module =
        analysis_module_alloc(100, ITERATED_ENSEMBLE_SMOOTHER);

    test_assert_true(analysis_module_set_var(
        std_module, ies::config::ENKF_TRUNCATION_KEY, "0.95"));
    test_assert_true(analysis_module_set_var(
        ies_module, ies::config::ENKF_TRUNCATION_KEY, "0.95"));

    analysis_module_init_update(std_module, testdata.ens_mask,
                                testdata.obs_mask, testdata.S, testdata.R,
                                testdata.dObs, testdata.E, testdata.D, rng);

    analysis_module_initX(std_module, X, prior, testdata.S, testdata.R,
                          testdata.dObs, testdata.E, testdata.D, rng);

    rng_free(rng);
    matrix_free(X);
    if (prior)
        matrix_free(prior);

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
    test_steplength1(path_testdata);
}
