#ifndef ERT_STD_ENKF_H
#define ERT_STD_ENKF_H

#include <ert/res_util/matrix.hpp>
#include <ert/util/rng.hpp>

void std_enkf_initX(void *module_data, matrix_type *X, const matrix_type *A,
                    const matrix_type *S, const matrix_type *R,
                    const matrix_type *dObs, const matrix_type *E,
                    const matrix_type *D, rng_type *rng);

#endif
