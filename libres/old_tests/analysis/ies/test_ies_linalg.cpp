#include <ert/util/rng.h>
#include <ert/util/test_util.hpp>

#include <ert/res_util/es_testdata.hpp>

#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_data.hpp>

void update_exact_scheme_subspace_no_truncation_diagR(
    const res::es_testdata &testdata, const ies::config::Config &ies_config,
    ies::data::Data &ies_data, Eigen::MatrixXd &A, rng_type *rng) {
    ies::init_update(ies_data, testdata.ens_mask, testdata.obs_mask);
    int iteration_nr = ies_data.inc_iteration_nr();
    ies::updateA(ies_data, A, testdata.S, testdata.R, testdata.E, testdata.D,
                 ies_config.inversion(), ies_config.truncation(),
                 ies_config.steplength(iteration_nr));
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
    Eigen::MatrixXd A1 = testdata.make_state("prior");
    Eigen::MatrixXd A2 = testdata.make_state("prior");

    ies::data::Data ies_data1(testdata.active_ens_size);
    ies::config::Config ies_config1(true);

    ies::data::Data ies_data2(testdata.active_ens_size);
    ies::config::Config ies_config2(true);

    ies_config1.truncation(1.0);
    ies_config1.max_steplength(0.6);
    ies_config1.min_steplength(0.6);
    ies_config1.inversion(ies::config::IES_INVERSION_SUBSPACE_EXACT_R);

    ies_config2.max_steplength(0.6);
    ies_config2.min_steplength(0.6);
    ies_config2.inversion(ies::config::IES_INVERSION_EXACT);

    update_exact_scheme_subspace_no_truncation_diagR(testdata, ies_config1,
                                                     ies_data1, A1, rng);
    update_exact_scheme_subspace_no_truncation_diagR(testdata, ies_config2,
                                                     ies_data2, A2, rng);
    test_assert_true(A1.isApprox(A2, 5e-5));
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
    Eigen::MatrixXd A1 = testdata.make_state("prior");
    Eigen::MatrixXd A2 = testdata.make_state("prior");

    ies::data::Data ies_data1(testdata.active_ens_size);
    ies::config::Config ies_config1(true);

    ies::data::Data ies_data2(testdata.active_ens_size);
    ies::config::Config ies_config2(true);

    ies_config1.truncation(0.95);
    ies_config1.max_steplength(0.6);
    ies_config1.min_steplength(0.6);
    ies_config1.inversion(ies::config::IES_INVERSION_SUBSPACE_EE_R);

    ies_config2.truncation(0.95);
    ies_config2.max_steplength(0.6);
    ies_config2.min_steplength(0.6);
    ies_config2.inversion(ies::config::IES_INVERSION_SUBSPACE_RE);

    update_exact_scheme_subspace_no_truncation_diagR(testdata, ies_config1,
                                                     ies_data1, A1, rng);
    update_exact_scheme_subspace_no_truncation_diagR(testdata, ies_config2,
                                                     ies_data2, A2, rng);
    test_assert_true(A1.isApprox(A2, 5e-6));

    rng_free(rng);
}

int main(int argc, char **argv) {
    res::es_testdata testdata(argv[1]);
    test_consistency_exact_scheme_subspace_no_truncation_diagR(testdata);
    test_consistency_scheme_inversions(testdata);
}
