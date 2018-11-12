#include <ert/util/test_util.hpp>
#include <ert/util/rng.h>

#include <ert/res_util/es_testdata.hpp>

#include <ert/analysis/std_enkf.hpp>
#include "ies_enkf_data.h"
#include "ies_enkf_config.h"
#include "ies_enkf.h"

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


void cmp_std_ies(const res::es_testdata& testdata) {
  rng_type * rng = rng_alloc( MZRAN, INIT_DEFAULT );
  matrix_type * A1 = testdata.alloc_state("prior");
  matrix_type * A2 = testdata.alloc_state("prior");
  matrix_type * X = matrix_alloc(testdata.active_ens_size, testdata.active_ens_size);

  ies_enkf_data_type * ies_data1 = static_cast<ies_enkf_data_type*>(ies_enkf_data_alloc(rng));
  ies_enkf_config_type * ies_config1 = ies_enkf_data_get_config(ies_data1);
  std_enkf_data_type * std_data = static_cast<std_enkf_data_type*>(std_enkf_data_alloc());

  ies_enkf_config_set_truncation(ies_config1, 0.95);
  ies_enkf_config_set_ies_min_steplength(ies_config1, 1.0);
  ies_enkf_config_set_ies_max_steplength(ies_config1, 1.0);
  ies_enkf_config_set_ies_inversion(ies_config1, IES_INVERSION_SUBSPACE_EXACT_R);
  ies_enkf_config_set_ies_aaprojection(ies_config1, false);

  std_enkf_set_truncation(std_data, 0.95);

  ies_enkf_init_update(ies_data1,
                       testdata.ens_mask,
                       testdata.obs_mask,
                       testdata.S,
                       testdata.R,
                       testdata.dObs,
                       testdata.E,
                       testdata.D,
                       rng);

  ies_enkf_updateA(ies_data1,
                   A1,
                   testdata.S,
                   testdata.R,
                   testdata.dObs,
                   testdata.E,
                   testdata.D,
                   NULL,
                   rng);

  std_enkf_initX(std_data,
                 X,
                 nullptr,
                 testdata.S,
                 testdata.R,
                 testdata.dObs,
                 testdata.E,
                 testdata.D,
                 rng);

  matrix_inplace_matmul(A2, X);
  test_assert_true( matrix_similar(A1, A2, 5e-6));

  matrix_free(A1);
  matrix_free(A2);
  std_enkf_data_free(std_data);
  ies_enkf_data_free(ies_data1);
  rng_free( rng );
}


int main(int argc, char ** argv) {
  res::es_testdata testdata(argv[1]);
  cmp_std_ies(testdata);
}
