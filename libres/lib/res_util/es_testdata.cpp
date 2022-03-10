/*
  Copyright (C) 2019  Equinor ASA, Norway.
  This file  is part of ERT - Ensemble based Reservoir Tool.

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

#include <iostream>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <filesystem>

#include <ert/res_util/matrix.hpp>

#include <ert/res_util/es_testdata.hpp>

namespace fs = std::filesystem;

#define ROW_MAJOR_STORAGE true

namespace res {

namespace {

class pushd {
public:
    pushd(const std::string &path, bool mkdir = false) {
        if (!util_is_directory(path.c_str())) {
            if (mkdir)
                util_make_path(path.c_str());
        }

        if (!util_is_directory(path.c_str()))
            throw std::invalid_argument("The path: " + path +
                                        " does not exist - can not proceed");

        this->org_cwd = util_alloc_cwd();
        util_chdir(path.c_str());
    }

    ~pushd() {
        util_chdir(this->org_cwd);
        free(this->org_cwd);
    }

private:
    char *org_cwd;
};

Eigen::MatrixXd load_matrix(const std::string &name, int rows, int columns) {
    if (!fs::exists(name))
        throw std::invalid_argument("File not found");

    FILE *stream = util_fopen(name.c_str(), "r");
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(rows, columns);
    matrix_fscanf_data(&m, ROW_MAJOR_STORAGE, stream);
    fclose(stream);

    return m;
}

void save_matrix_data(const std::string &name, const Eigen::MatrixXd &m) {
    FILE *stream = util_fopen(name.c_str(), "w");
    matrix_fprintf_data(&m, ROW_MAJOR_STORAGE, stream);
    fclose(stream);
}

std::array<int, 2> load_size() {
    int active_ens_size, active_obs_size;

    FILE *stream = fopen("size", "r");
    if (!stream)
        throw std::invalid_argument(
            "Could not find file: size with ens_size, obs_size information in "
            "the test directory.");

    int read_count =
        fscanf(stream, "%d %d", &active_ens_size, &active_obs_size);
    if (read_count != 2)
        throw std::invalid_argument(
            "Failed to read ens_size obs_size from size file");

    fclose(stream);

    return {active_ens_size, active_obs_size};
}

void save_size(int ens_size, int obs_size) {
    FILE *stream = util_fopen("size", "w");
    fprintf(stream, "%d %d\n", ens_size, obs_size);
    fclose(stream);
}

void matrix_delete_row_column(Eigen::MatrixXd &m1, int row_column) {
    matrix_delete_row(&m1, row_column);
    matrix_delete_column(&m1, row_column);
}

} // namespace

Eigen::MatrixXd es_testdata::make_matrix(const std::string &fname, int rows,
                                         int columns) const {
    pushd tmp_path(this->path);
    return load_matrix(fname, rows, columns);
}

void es_testdata::save_matrix(const std::string &name,
                              const Eigen::MatrixXd &m) const {
    pushd tmp_path(this->path);

    FILE *stream = util_fopen(name.c_str(), "w");
    matrix_fprintf_data(&m, ROW_MAJOR_STORAGE, stream);
    fclose(stream);
}

es_testdata::es_testdata(const Eigen::MatrixXd &S, const Eigen::MatrixXd &R,
                         const Eigen::MatrixXd &D, const Eigen::MatrixXd &E,
                         const Eigen::MatrixXd &dObs)
    : S(S), R(R), D(D), E(E), dObs(dObs), active_ens_size(S.cols()),
      active_obs_size(S.rows()),
      obs_mask(std::vector<bool>(active_obs_size, true)),
      ens_mask(std::vector<bool>(active_ens_size, true)) {}

void es_testdata::deactivate_obs(int iobs) {
    if (iobs >= this->obs_mask.size())
        throw std::invalid_argument("Obs number: " + std::to_string(iobs) +
                                    " out of reach");

    if (this->obs_mask[iobs]) {
        this->obs_mask[iobs] = false;

        matrix_delete_row(&this->dObs, iobs);
        matrix_delete_row(&this->S, iobs);
        matrix_delete_row_column(this->R, iobs);
        matrix_delete_row(&this->E, iobs);
        matrix_delete_row(&this->D, iobs);

        this->active_obs_size -= 1;
    }
}

void es_testdata::deactivate_realization(int iens) {
    if (iens >= this->ens_mask.size())
        throw std::invalid_argument(
            "iRealization number: " + std::to_string(iens) + " out of reach");

    if (this->ens_mask[iens]) {
        this->ens_mask[iens] = false;

        matrix_delete_column(&this->S, iens);
        matrix_delete_column(&this->E, iens);
        matrix_delete_column(&this->D, iens);

        this->active_ens_size -= 1;
    }
}

es_testdata::es_testdata(const char *path) : path(path) {
    pushd tmp_path(this->path);
    auto size = load_size();
    this->active_ens_size = size[0];
    this->active_obs_size = size[1];

    S = load_matrix("S", this->active_obs_size, this->active_ens_size);
    E = load_matrix("E", this->active_obs_size, this->active_ens_size);
    R = load_matrix("R", this->active_obs_size, this->active_obs_size);
    D = load_matrix("D", this->active_obs_size, this->active_ens_size);
    dObs = load_matrix("dObs", this->active_obs_size, 2);

    this->obs_mask = std::vector<bool>(this->active_obs_size, true);
    this->ens_mask = std::vector<bool>(this->active_ens_size, true);
}

void es_testdata::save(const std::string &path) const {
    pushd tmp_path(path, true);
    save_size(this->active_ens_size, this->active_obs_size);

    save_matrix_data("S", this->S);
    save_matrix_data("E", this->E);
    save_matrix_data("R", this->R);
    save_matrix_data("D", this->D);
    save_matrix_data("dObs", this->dObs);
}

/*
  This function will allocate a matrix based on data found on disk. The data on
  disk is only the actual content of the matrix, in row_major order. Before the
  matrix is constructed it is verified that the number of elements is a multiple
  of this->active_ens_size.
*/

Eigen::MatrixXd es_testdata::make_state(const std::string &name) const {

    pushd tmp_path(this->path);

    std::ifstream stream(name);
    if (!stream)
        throw std::invalid_argument("No such state matrix: " + this->path +
                                    "/" + name);
    std::istream_iterator<double> start(stream), end;
    std::vector<double> data(start, end);

    if ((data.size() % this->active_ens_size) != 0)
        throw std::invalid_argument(
            "Number of elements in file with state informaton must be a "
            "multiple of ensemble_size: " +
            std::to_string(this->active_ens_size));

    int state_size = data.size() / this->active_ens_size;
    Eigen::MatrixXd state =
        Eigen::MatrixXd::Zero(state_size, this->active_ens_size);
    for (int is = 0; is < state_size; is++) {
        for (int iens = 0; iens < this->active_ens_size; iens++) {
            state(is, iens) = data[iens + is * this->active_ens_size];
        }
    }

    return state;
}

} // namespace res
