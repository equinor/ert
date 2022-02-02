#include <ert/util/test_util.hpp>
#include <ert/util/rng.h>

#include <ert/res_util/es_testdata.hpp>

#include <ert/analysis/std_enkf.hpp>
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

    auto *ies_data1 =
        static_cast<ies::data::data_type *>(ies::data::alloc(true));
    auto *ies_config1 = ies::data::get_config(ies_data1);
    auto *std_data =
        static_cast<ies::data::data_type *>(ies::data::alloc(false));
    auto *std_config = ies::data::get_config(std_data);

    ies::config::set_truncation(ies_config1, 0.95);
    ies::config::set_min_steplength(ies_config1, 1.0);
    ies::config::set_max_steplength(ies_config1, 1.0);
    ies::config::set_inversion(ies_config1,
                               ies::config::IES_INVERSION_SUBSPACE_EXACT_R);
    ies::config::set_aaprojection(ies_config1, false);

    ies::config::set_truncation(std_config, 0.95);

    ies::init_update(ies_data1, testdata.ens_mask, testdata.obs_mask,
                     testdata.S, testdata.R, testdata.dObs, testdata.E,
                     testdata.D, rng);

    ies::updateA(ies_data1, A1, testdata.S, testdata.R, testdata.dObs,
                 testdata.E, testdata.D, rng);

    std_enkf_initX(std_data, X, nullptr, testdata.S, testdata.R, testdata.dObs,
                   testdata.E, testdata.D, rng);

    matrix_inplace_matmul(A2, X);
    test_assert_true(matrix_similar(A1, A2, 5e-6));

    matrix_free(A1);
    matrix_free(A2);
    ies::data::free(std_data);
    ies::data::free(ies_data1);
    rng_free(rng);
}

int main(int argc, char **argv) {
    res::es_testdata testdata(argv[1]);
    cmp_std_ies(testdata);
}
