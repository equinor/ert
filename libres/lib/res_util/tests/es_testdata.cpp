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
        matrix_iset(E, i, j, v);
        matrix_iset(D, i, j, v);
        v += 1;
      }
    }

    v = 0;
    for (int i=0; i < matrix_get_rows(dObs); i++) {
      for (int j=0; j < matrix_get_columns(dObs); j++) {
        matrix_iset(dObs, i, j, v);
        v += 1;
      }
    }

    v = 0;
    for (int i=0; i < matrix_get_rows(R); i++) {
      for (int j=0; j < matrix_get_columns(R); j++) {
        matrix_iset(R, i, j, v);
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

void test_deactivate() {
  int ens_size = 10;
  int obs_size =  7;
  int del_iobs = 3;
  int del_iens = 7;
  res::es_testdata td1 = make_testdata(ens_size, obs_size);
  res::es_testdata td2 = make_testdata(ens_size, obs_size);



  test_assert_throw(td1.deactivate_obs(10), std::invalid_argument);
  td1.deactivate_obs(del_iobs);

  for (int i1=0; i1 < matrix_get_rows(td1.R); i1++) {
    int i2 = i1 + (i1 >= del_iobs ? 1 : 0);
    for (int j1=0; j1 < matrix_get_columns(td1.R); j1++) {
      int j2 = j1 + (j1 >= del_iobs ? 1 : 0);

      test_assert_double_equal(matrix_iget(td1.R, i1, j1), matrix_iget(td2.R, i2, j2));
    }

    test_assert_double_equal(matrix_iget(td1.dObs, i1, 0), matrix_iget(td2.dObs, i2, 0));
    test_assert_double_equal(matrix_iget(td1.dObs, i1, 1), matrix_iget(td2.dObs, i2, 1));
    for (int j1=0; j1 < matrix_get_columns(td1.S); j1++) {
      int j2 = j1;
      test_assert_double_equal(matrix_iget(td1.S, i1, j1), matrix_iget(td2.S, i2, j2));

      if (td1.E)
        test_assert_double_equal(matrix_iget(td1.E, i1, j1), matrix_iget(td2.E, i2, j2));

      if (td1.D)
        test_assert_double_equal(matrix_iget(td1.D, i1, j1), matrix_iget(td2.D, i2, j2));
    }
  }

  td1.deactivate_realization(del_iens);

  for (int i1=0; i1 < matrix_get_rows(td1.S); i1++) {
    int i2 = i1 + (i1 >= del_iobs ? 1 : 0);

    for (int j1=0; j1 < matrix_get_columns(td1.S); j1++) {
      int j2 = j1 + (j1 >= del_iens ? 1 : 0);

      test_assert_double_equal(matrix_iget(td1.S, i1, j1), matrix_iget(td2.S, i2, j2));

      if (td1.E)
        test_assert_double_equal(matrix_iget(td1.E, i1, j1), matrix_iget(td2.E, i2, j2));

      if (td1.D)
        test_assert_double_equal(matrix_iget(td1.D, i1, j1), matrix_iget(td2.D, i2, j2));
    }
  }

  test_assert_int_equal(td1.active_ens_size, ens_size - 1);
  test_assert_int_equal(td1.active_obs_size, obs_size - 1);
}


void test_basic() {
  ecl::util::TestArea work_area("es_testdata");
  int ens_size = 10;
  int obs_size =  7;
  res::es_testdata td1 = make_testdata(ens_size, obs_size);
  td1.save("path/sub/path");
  res::es_testdata td2("path/sub/path");
  test_assert_true( matrix_equal(td1.S, td2.S) );

  test_assert_int_equal( bool_vector_size(td1.obs_mask), obs_size );
  test_assert_int_equal( bool_vector_count_equal(td1.obs_mask, true), obs_size );

  test_assert_int_equal( bool_vector_size(td1.ens_mask), ens_size );
  test_assert_int_equal( bool_vector_count_equal(td1.ens_mask, true), ens_size );
}


void test_size_problems() {
  ecl::util::TestArea work_area("es_testdata");
  int ens_size = 10;
  int obs_size =  7;
  {
    res::es_testdata td1 = make_testdata(ens_size, obs_size);
    td1.save("path");
  }
  unlink("path/size");
  test_assert_throw( res::es_testdata("path"), std::invalid_argument );
  {
    FILE * fp = util_fopen("path/size", "w");
    fprintf(fp, "%d\n", ens_size);
    fclose(fp);
  }
  test_assert_throw( res::es_testdata("path"), std::invalid_argument );
}

void test_load_state() {
  ecl::util::TestArea work_area("es_testdata");
  int ens_size = 10;
  int obs_size =  7;
  res::es_testdata td0 = make_testdata(ens_size, obs_size);
  td0.save("PATH");

  res::es_testdata td("PATH");
  td.deactivate_realization(5);
  int active_ens_size = td.active_ens_size;

  test_assert_throw( td.alloc_state("DOES_NOT_EXIST"), std::invalid_argument);

  {
    int invalid_size = 10 * active_ens_size + active_ens_size / 2;
    FILE * stream = util_fopen("PATH/A0", "w");
    for (int i = 0; i < invalid_size; i++)
      fprintf(stream, "%d\n", i);
    fclose(stream);
    test_assert_throw( td.alloc_state("A0"), std::invalid_argument);
  }
  {
    int state_size = 7;
    int valid_size = state_size * active_ens_size;
    FILE * stream = util_fopen("PATH/A1", "w");
    double value = 0;
    for (int row=0; row < state_size; row++) {
      for (int iens = 0; iens < active_ens_size; iens++) {
        fprintf(stream, "%lg ", value);
        value++;
      }
      fputs("\n", stream);
    }
    fclose(stream);

    matrix_type * A1 = td.alloc_state("A1");
    test_assert_int_equal(matrix_get_rows(A1), state_size);
    test_assert_int_equal(matrix_get_columns(A1), active_ens_size);

    value = 0;
    for (int row=0; row < state_size; row++) {
      for (int iens = 0; iens < active_ens_size; iens++) {
        test_assert_double_equal(matrix_iget(A1, row, iens), value);
        value++;
      }
    }

    td.save_matrix("A2", A1);
    matrix_type * A2 = td.alloc_matrix("A2", state_size, active_ens_size);
    test_assert_true( matrix_equal(A1,A2) );

    matrix_free(A1);
    matrix_free(A2);
  }
}

int main() {
  test_basic();
  test_load_state();
  test_deactivate();
  test_size_problems();
}
