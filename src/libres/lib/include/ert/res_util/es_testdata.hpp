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

#include <ert/util/bool_vector.hpp>

#include <ert/res_util/matrix.hpp>

namespace res {
class es_testdata {
public:
  std::string path;

  matrix_type * S;
  matrix_type * E;
  matrix_type * R;
  matrix_type * D;
  matrix_type * dObs;
  int active_obs_size;
  int active_ens_size;
  bool_vector_type * obs_mask;
  bool_vector_type * ens_mask;
  int state_size;

  es_testdata(const matrix_type* S, const matrix_type * R, const matrix_type * dObs, const matrix_type *D , const matrix_type * E);
  es_testdata(const char * path);
  ~es_testdata();

  matrix_type * alloc_matrix(const std::string& name, int rows, int columns) const;
  void save_matrix(const std::string& name, const matrix_type * m) const;
  matrix_type * alloc_state(const std::string& name) const;
  void save(const std::string& path) const;
  void deactivate_obs(int iobs);
  void deactivate_realization(int iens);
};

}

#endif
