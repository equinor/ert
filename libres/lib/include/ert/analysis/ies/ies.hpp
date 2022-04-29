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

#include <Eigen/Dense>
#include <ert/analysis/ies/ies_config.hpp>
#include <ert/analysis/ies/ies_data.hpp>

namespace ies {

void linalg_store_active_W(data::Data *data, const Eigen::MatrixXd &W0);

Eigen::MatrixXd make_activeE(const data::Data *data);
Eigen::MatrixXd make_activeW(const data::Data *data);
Eigen::MatrixXd make_activeA(const data::Data *data);

void init_update(data::Data &module_data, const std::vector<bool> &ens_mask,
                 const std::vector<bool> &obs_mask);

Eigen::MatrixXd makeX(const Eigen::MatrixXd &A, const Eigen::MatrixXd &Y0,
                      const Eigen::MatrixXd &R, const Eigen::MatrixXd &E,
                      const Eigen::MatrixXd &D,
                      const ies::config::inversion_type ies_inversion,
                      const std::variant<double, int> &truncation,
                      Eigen::MatrixXd &W0, double ies_steplength,
                      int iteration_nr);

void updateA(data::Data &data,
             // Updated ensemble A returned to ERT.
             Eigen::Ref<Eigen::MatrixXd> A,
             // Ensemble of predicted measurements
             const Eigen::MatrixXd &Yin,
             // Measurement error covariance matrix (not used)
             const Eigen::MatrixXd &Rin,
             // Ensemble of observation perturbations
             const Eigen::MatrixXd &Ein,
             // (d+E-Y) Ensemble of perturbed observations - Y
             const Eigen::MatrixXd &Din,
             const ies::config::inversion_type ies_inversion,
             const std::variant<double, int> &truncation,
             double ies_steplength);

Eigen::MatrixXd makeE(const Eigen::VectorXd &obs_errors,
                      const Eigen::MatrixXd &noise);
Eigen::MatrixXd makeD(const Eigen::VectorXd &obs_values,
                      const Eigen::MatrixXd &E, const Eigen::MatrixXd &S);
} // namespace ies

#endif
