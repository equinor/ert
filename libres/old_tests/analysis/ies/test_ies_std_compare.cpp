#include <ert/util/test_util.hpp>
#include <ert/util/rng.h>

#include <ert/res_util/es_testdata.hpp>

#include <ert/analysis/ies/ies_data.hpp>
#include <ert/analysis/ies/ies.hpp>

/*
TEST 3 (consistency between IES and STD_ENKF):
 - ANALYSIS_SET_VAR IES_ENKF ENKF_TRUNCATION         0.999
 - ANALYSIS_SET_VAR IES_ENKF IES_STEPLENGTH          1.0
 - ANALYSIS_SET_VAR IES_ENKF IES_INVERSION           1
 - ANALYSIS_SET_VAR IES_ENKF IES_AAPROJECTION        false
should give same result as:
 - ANALYSIS_SET_VAR IES_ENKF ENKF_TRUNCATION         0.999
 - ANALYSIS_SELECT STD_ENKF
*/

void cmp_std_ies(const res::es_testdata &testdata) {
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    matrix_type *A1 = testdata.alloc_state("prior");
    matrix_type *A2 = testdata.alloc_state("prior");
    matrix_type *X =
        matrix_alloc(testdata.active_ens_size, testdata.active_ens_size);
    ies::data::Data ies_data1(testdata.active_ens_size, true);
    ies::data::Data std_data(testdata.active_ens_size, false);

    auto &ies_config1 = ies_data1.config();
    auto &std_config = std_data.config();

    ies_config1.truncation(0.95);
    ies_config1.min_steplength(1.0);
    ies_config1.max_steplength(1.0);
    ies_config1.inversion(ies::config::IES_INVERSION_SUBSPACE_EXACT_R);
    ies_config1.aaprojection(false);

    std_config.truncation(0.95);

    ies::init_update(&ies_data1, testdata.ens_mask, testdata.obs_mask,
                     testdata.S, testdata.R, testdata.E, testdata.D);

    ies::updateA(&ies_data1, A1, testdata.S, testdata.R, testdata.E,
                 testdata.D);

    ies::initX(&std_data, testdata.S, testdata.R, testdata.E, testdata.D, X);

    matrix_inplace_matmul(A2, X);
    test_assert_true(matrix_similar(A1, A2, 5e-6));

    matrix_free(A1);
    matrix_free(A2);
    rng_free(rng);
}

int main(int argc, char **argv) {
    res::es_testdata testdata(argv[1]);
    cmp_std_ies(testdata);
}
