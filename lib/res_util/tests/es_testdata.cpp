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
#include <stdexcept>

#include <ert/res_util/es_testdata.hpp>
#include <ert/util/test_work_area.hpp>
#include <ert/util/test_util.hpp>



res::es_testdata make_testdata(int ens_size, int obs_size) {
  matrix_type * S = matrix_alloc(obs_size, ens_size);
  matrix_type * E = matrix_alloc(obs_size, ens_size);
  matrix_type * R = matrix_alloc(obs_size, obs_size);
  matrix_type * D = matrix_alloc(obs_size, ens_size);
  matrix_type * dObs = matrix_alloc(obs_size, 2);

  {
    double v = 0;
    for (int i=0; i < matrix_get_rows(S); i++) {
      for (int j=0; j < matrix_get_columns(S); j++) {
        matrix_iset(S, i, j, v);
        v += 1;
      }
    }
  }
  res::es_testdata td(S,R,dObs,D,E);

  matrix_free(S);
  matrix_free(R);
  matrix_free(dObs);
  matrix_free(D);
  matrix_free(E);

  return td;
}


void test_basic() {
  test_work_area_type * work_area = test_work_area_alloc("es_testdata");

  int ens_size = 10;
  int obs_size =  7;
  res::es_testdata td1 = make_testdata(ens_size, obs_size);
  td1.save("path/sub/path");

  res::es_testdata td2("path/sub/path");
  test_assert_true( matrix_equal(td1.S, td2.S) );

  test_work_area_free(work_area);
}


void test_load_state() {
  test_work_area_type * work_area = test_work_area_alloc("es_testdata");
  int ens_size = 10;
  int obs_size =  7;
  res::es_testdata td0 = make_testdata(ens_size, obs_size);
  td0.save("PATH");

  res::es_testdata td("PATH");


  test_assert_throw( td.alloc_state("DOES_NOT_EXIST"), std::invalid_argument);

  {
    int invalid_size = 10 * ens_size + ens_size / 2;
    FILE * stream = util_fopen("PATH/A0", "w");
    for (int i = 0; i < invalid_size; i++)
      fprintf(stream, "%d\n", i);
    fclose(stream);
    test_assert_throw( td.alloc_state("A0"), std::invalid_argument);
  }
  {
    int state_size = 7;
    int valid_size = state_size * ens_size;
    FILE * stream = util_fopen("PATH/A1", "w");
    double value = 0;
    for (int row=0; row < state_size; row++) {
      for (int iens = 0; iens < ens_size; iens++) {
        fprintf(stream, "%lg ", value);
        value++;
      }
      fputs("\n", stream);
    }
    fclose(stream);

    matrix_type * A1 = td.alloc_state("A1");
    test_assert_int_equal(matrix_get_rows(A1), state_size);
    test_assert_int_equal(matrix_get_columns(A1), ens_size);

    value = 0;
    for (int row=0; row < state_size; row++) {
      for (int iens = 0; iens < ens_size; iens++) {
        test_assert_double_equal(matrix_iget(A1, row, iens), value);
        value++;
      }
    }

    td.save_matrix("A2", A1);
    matrix_type * A2 = td.alloc_matrix("A2", state_size, ens_size);
    test_assert_true( matrix_equal(A1,A2) );

    matrix_free(A1);
    matrix_free(A2);
  }
  test_work_area_free(work_area);
}

int main() {
  test_basic();
  test_load_state();
}
