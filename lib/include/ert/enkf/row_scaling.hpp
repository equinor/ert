/*
  Copyright (C) 2020  Equinor ASA, Norway.

  The file 'row_scaling.hpp' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ROW_SCALING_H
#define ROW_SCALING_H

#include <ert/res_util/matrix.hpp>

class row_scaling {
public:
  double operator[](int index) const;
  void assign(int index, double value);
  void multiply(matrix_type * A, const matrix_type * X0) const;
  int size() const;
private:
  int resolution = 1000;
  std::vector<double> data;
};

typedef row_scaling row_scaling_type;

#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/type_macros.h>


row_scaling_type * row_scaling_alloc();
row_scaling_type * row_scaling_alloc_copy(const row_scaling_type * row_scaling);
void               row_scaling_free(row_scaling_type * row_scaling);
void               row_scaling_multiply(const row_scaling_type * row_scaling, matrix_type * A, const matrix_type * X0);
int                row_scaling_get_size(const row_scaling_type * row_scaling);
double             row_scaling_iget(const row_scaling_type * row_scaling, int index);
void               row_scaling_iset(row_scaling_type * row_scaling, int index, double value);

UTIL_IS_INSTANCE_HEADER( row_scaling );


#ifdef __cplusplus
}
#endif
#endif


