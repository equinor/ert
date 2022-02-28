/*
  Copyright (C) 2019  Equinor ASA, Norway.

  The file 'ies_enkf.hpp' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef IES_ENKF_H
#define IES_ENKF_H
#include <variant>

#include <ert/util/bool_vector.hpp>
#include <ert/util/rng.hpp>

#include <ert/res_util/matrix.hpp>
#include <ert/analysis/ies/ies_data.hpp>

namespace ies {

void linalg_store_active_W(data::Data *data, const matrix_type *W0);

matrix_type *alloc_activeE(const data::Data *data);
matrix_type *alloc_activeW(const data::Data *data);
matrix_type *alloc_activeA(const data::Data *data);

void init_update(data::Data &module_data, const std::vector<bool> &ens_mask,
                 const std::vector<bool> &obs_mask, const matrix_type *S,
                 const matrix_type *R, const matrix_type *E,
                 const matrix_type *D);

/*
  Internally in the ies algorithm there is a standard ES update step, that
  functionality is exported so that it can be utilised in ES updates which are
  not iterative.
*/
void initX(const config::Config &ies_config, const matrix_type *S,
           const matrix_type *R, const matrix_type *E, const matrix_type *D,
           matrix_type *X);

void updateA(
    const config::Config &ies_config, data::Data &data,
    matrix_type *A,          // Updated ensemble A returned to ERT.
    const matrix_type *Yin,  // Ensemble of predicted measurements
    const matrix_type *Rin,  // Measurement error covariance matrix (not used)
    const matrix_type *Ein,  // Ensemble of observation perturbations
    const matrix_type *Din); // (d+E-Y) Ensemble of perturbed observations - Y
} // namespace ies

#endif
