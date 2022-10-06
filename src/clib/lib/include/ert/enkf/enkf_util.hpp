#ifndef ERT_ENKF_UTIL_H
#define ERT_ENKF_UTIL_H
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <ert/util/buffer.h>
#include <ert/util/rng.h>

#include <ert/ecl/ecl_type.h>
#include <ert/ecl/ecl_util.h>

#include <ert/enkf/enkf_types.hpp>

double enkf_util_rand_normal(double, double, rng_type *rng);
void enkf_util_assert_buffer_type(buffer_type *buffer,
                                  ert_impl_type target_type);

#endif
