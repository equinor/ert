/*
  Copyright (C) 2019  Equinor ASA, Norway.

  The file 'analysis_test_external_module.c' is part of ERT - Ensemble based Reservoir Tool.

  ERT is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
  for more details.
*/

#include <stdlib.h>

#include <stdexcept>

#include <ert/util/test_util.hpp>
#include <ert/util/rng.hpp>

#include <ert/res_util/matrix.hpp>

#include <ert/analysis/analysis_module.hpp>



void test_invalid_mask_size() {
  analysis_module_type * module = analysis_module_alloc_internal("STD_ENKF");
  int active_ens_size = 10;
  int active_obs_size = 5;
  rng_type * rng = NULL;

  matrix_type * S = matrix_alloc(active_obs_size, active_ens_size);
  matrix_type * R = matrix_alloc(active_obs_size, active_obs_size);
  matrix_type * dObs = matrix_alloc(active_obs_size, 2);
  matrix_type * E = matrix_alloc(active_obs_size, active_obs_size);
  matrix_type * D = matrix_alloc(active_obs_size, active_obs_size);

  bool_vector_type * ens_mask = bool_vector_alloc(active_ens_size + 1, true );
  bool_vector_type * obs_mask = bool_vector_alloc(active_obs_size + 1, true );

  test_assert_throw( analysis_module_init_update(module,
                                                 ens_mask,
                                                 obs_mask,
                                                 S,
                                                 R,
                                                 dObs,
                                                 E,
                                                 D,
                                                 rng),
                     std::invalid_argument);

  bool_vector_iset(ens_mask, 0, false);
  bool_vector_iset(obs_mask, 0, false);

  analysis_module_init_update(module,
                              ens_mask,
                              obs_mask,
                              S,
                              R,
                              dObs,
                              E,
                              D,
                              rng);

  bool_vector_free(ens_mask);
  bool_vector_free(obs_mask);

  matrix_free(S);
  matrix_free(R);
  matrix_free(dObs);
  matrix_free(E);
  matrix_free(D);

  analysis_module_free( module );
}



int main(int argc, char **argv) {
  test_invalid_mask_size();
}
