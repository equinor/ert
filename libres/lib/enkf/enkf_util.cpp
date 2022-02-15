/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_util.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <cmath>
#include <stdlib.h>
#include <random>
#include <ert/util/util.h>
#include <ert/util/rng.h>
#include <ert/ecl/ecl_util.h>

#include <ert/enkf/enkf_util.hpp>

class generator {
    rng_type *rng;

public:
    generator(rng_type *rng) : rng(rng) {}

    using value_type = unsigned int;
    static constexpr value_type min() { return 0; }
    static constexpr value_type max() { return UINT32_MAX; }

    value_type operator()() { return rng_forward(rng); }
};

double enkf_util_rand_normal(double mean, double std, rng_type *rng) {
    generator gen(rng);
    std::normal_distribution<double> normdist{mean, std};
    return normdist(gen);
}

void enkf_util_assert_buffer_type(buffer_type *buffer,
                                  ert_impl_type target_type) {
    ert_impl_type file_type = INVALID;
    file_type = (ert_impl_type)buffer_fread_int(buffer);
    if (file_type != target_type)
        util_abort(
            "%s: wrong target type in file (expected:%d  got:%d) - aborting \n",
            __func__, target_type, file_type);
}
