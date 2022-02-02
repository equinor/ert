/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'std_enkf.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/res_util/matrix.hpp>

#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_config.hpp>

void std_enkf_initX(void *module_data, matrix_type *X, const matrix_type *A,
                    const matrix_type *S, const matrix_type *R,
                    const matrix_type *dObs, const matrix_type *E,
                    const matrix_type *D, rng_type *rng) {

    auto *data = ies::data::data_safe_cast(module_data);
    ies::initX(data, S, R, E, D, X);
}
