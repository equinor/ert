/*
  Copyright (C) 2019  Equinor ASA, Norway.
  This file is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ES_TESTDATA_HPP
#define ES_TESTDATA_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <ert/util/bool_vector.hpp>

namespace res {
class es_testdata {
public:
    std::string path;

    Eigen::MatrixXd S{};
    Eigen::MatrixXd E{};
    Eigen::MatrixXd R{};
    Eigen::MatrixXd D{};
    Eigen::MatrixXd dObs{};
    int active_obs_size{};
    int active_ens_size{};
    std::vector<bool> obs_mask;
    std::vector<bool> ens_mask;
    int state_size{};

    es_testdata(const Eigen::MatrixXd &S, const Eigen::MatrixXd &R,
                const Eigen::MatrixXd &D, const Eigen::MatrixXd &E,
                const Eigen::MatrixXd &dObs);
    es_testdata(const char *path);

    Eigen::MatrixXd make_matrix(const std::string &name, int rows,
                                int columns) const;
    void save_matrix(const std::string &name, const Eigen::MatrixXd &m) const;
    Eigen::MatrixXd make_state(const std::string &name) const;
    void save(const std::string &path) const;
    void deactivate_obs(int iobs);
    void deactivate_realization(int iens);
};

} // namespace res

#endif
