#include <ert/util/test_util.hpp>
#include <ert/util/rng.h>

#include <ert/res_util/es_testdata.hpp>

#include <ert/analysis/ies/ies_data.hpp>
#include <ert/analysis/ies/ies.hpp>

void update_exact_scheme_subspace_no_truncation_diagR(
    const res::es_testdata &testdata, ies::data::data_type *ies_data,
    matrix_type *A, rng_type *rng) {
    ies::init_update(ies_data, testdata.ens_mask, testdata.obs_mask, testdata.S,
                     testdata.R, testdata.dObs, testdata.E, testdata.D, rng);

    ies::updateA(ies_data, A, testdata.S, testdata.R, testdata.dObs, testdata.E,
                 testdata.D, rng);
}

/*
TEST 1 (Consistency between exact scheme and subspace scheme with no truncation and exact diagonal R):
 - ANALYSIS_SET_VAR IES_ENKF ENKF_TRUNCATION         1.0
 - ANALYSIS_SET_VAR IES_ENKF IES_STEPLENGTH          0.6
 - ANALYSIS_SET_VAR IES_ENKF IES_INVERSION           1
 - ANALYSIS_SELECT IES_ENKF
should give same result as:
 - ANALYSIS_SET_VAR IES_ENKF IES_STEPLENGTH          0.6
 - ANALYSIS_SET_VAR IES_ENKF IES_INVERSION           0
 - ANALYSIS_SELECT IES_ENKF

*/

void test_consistency_exact_scheme_subspace_no_truncation_diagR(
    const res::es_testdata &testdata) {
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    matrix_type *A1 = testdata.alloc_state("prior");
    matrix_type *A2 = testdata.alloc_state("prior");

    auto *ies_data1 = static_cast<ies::data::data_type *>(
        ies::data::alloc(testdata.active_ens_size, true));
    auto &ies_config1 = ies::data::get_config(ies_data1);

    auto *ies_data2 = static_cast<ies::data::data_type *>(
        ies::data::alloc(testdata.active_ens_size, true));
    auto &ies_config2 = ies::data::get_config(ies_data2);

    ies_config1.truncation(1.0);
    ies_config1.max_steplength(0.6);
    ies_config1.min_steplength(0.6);
    ies_config1.inversion(ies::config::IES_INVERSION_SUBSPACE_EXACT_R);

    ies_config2.max_steplength(0.6);
    ies_config2.min_steplength(0.6);
    ies_config2.inversion(ies::config::IES_INVERSION_EXACT);

    update_exact_scheme_subspace_no_truncation_diagR(testdata, ies_data1, A1,
                                                     rng);
    update_exact_scheme_subspace_no_truncation_diagR(testdata, ies_data2, A2,
                                                     rng);
    test_assert_true(matrix_similar(A1, A2, 5e-5));

    matrix_free(A1);
    matrix_free(A2);
    ies::data::free(ies_data1);
    ies::data::free(ies_data2);
    rng_free(rng);
}

/*
TEST 2 (consistency between subspace inversion schemes with lowrank R):
 - ANALYSIS_SET_VAR IES_ENKF ENKF_TRUNCATION         0.999
 - ANALYSIS_SET_VAR IES_ENKF IES_STEPLENGTH          0.6
 - ANALYSIS_SET_VAR IES_ENKF IES_INVERSION           2
 - ANALYSIS_SELECT IES_ENKF
should give same result as
 - ANALYSIS_SET_VAR IES_ENKF ENKF_TRUNCATION         0.999
 - ANALYSIS_SET_VAR IES_ENKF IES_STEPLENGTH          0.6
 - ANALYSIS_SET_VAR IES_ENKF IES_INVERSION           3
 - ANALYSIS_SELECT IES_ENKF
*/

void test_consistency_scheme_inversions(const res::es_testdata &testdata) {
    rng_type *rng = rng_alloc(MZRAN, INIT_DEFAULT);
    matrix_type *A1 = testdata.alloc_state("prior");
    matrix_type *A2 = testdata.alloc_state("prior");

    auto *ies_data1 = static_cast<ies::data::data_type *>(
        ies::data::alloc(testdata.active_ens_size, true));
    auto &ies_config1 = ies::data::get_config(ies_data1);

    auto *ies_data2 = static_cast<ies::data::data_type *>(
        ies::data::alloc(testdata.active_ens_size, true));
    auto &ies_config2 = ies::data::get_config(ies_data2);

    ies_config1.truncation(0.95);
    ies_config1.max_steplength(0.6);
    ies_config1.min_steplength(0.6);
    ies_config1.inversion(ies::config::IES_INVERSION_SUBSPACE_EE_R);

    ies_config2.truncation(0.95);
    ies_config2.max_steplength(0.6);
    ies_config2.min_steplength(0.6);
    ies_config2.inversion(ies::config::IES_INVERSION_SUBSPACE_RE);

    update_exact_scheme_subspace_no_truncation_diagR(testdata, ies_data1, A1,
                                                     rng);
    update_exact_scheme_subspace_no_truncation_diagR(testdata, ies_data2, A2,
                                                     rng);
    test_assert_true(matrix_similar(A1, A2, 5e-6));

    matrix_free(A1);
    matrix_free(A2);
    ies::data::free(ies_data1);
    ies::data::free(ies_data2);
    rng_free(rng);
}

int main(int argc, char **argv) {
    res::es_testdata testdata(argv[1]);
    test_consistency_exact_scheme_subspace_no_truncation_diagR(testdata);
    test_consistency_scheme_inversions(testdata);
}
