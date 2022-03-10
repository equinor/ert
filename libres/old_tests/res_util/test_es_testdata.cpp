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
#include <algorithm>
#include <stdexcept>

#include <ert/res_util/es_testdata.hpp>
#include <ert/util/test_work_area.hpp>
#include <ert/util/test_util.hpp>

res::es_testdata make_testdata(int ens_size, int obs_size) {
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(obs_size, ens_size);
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(obs_size, ens_size);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(obs_size, obs_size);
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(obs_size, ens_size);
    Eigen::MatrixXd dObs = Eigen::MatrixXd::Zero(obs_size, 2);

    {
        double v = 0;
        for (int i = 0; i < S.rows(); i++) {
            for (int j = 0; j < S.cols(); j++) {
                S(i, j) = v;
                E(i, j) = v;
                D(i, j) = v;
                v += 1;
            }
        }
        v = 0;
        for (int i = 0; i < dObs.rows(); i++) {
            for (int j = 0; j < dObs.cols(); j++) {
                dObs(i, j) = v;
                v += 1;
            }
        }

        v = 0;
        for (int i = 0; i < R.rows(); i++) {
            for (int j = 0; j < R.cols(); j++) {
                R(i, j) = v;
                v += 1;
            }
        }
    }
    res::es_testdata td(S, R, D, E, dObs);

    return td;
}

void test_deactivate() {
    int ens_size = 10;
    int obs_size = 7;
    int del_iobs = 3;
    int del_iens = 7;
    res::es_testdata td1 = make_testdata(ens_size, obs_size);
    res::es_testdata td2 = make_testdata(ens_size, obs_size);

    test_assert_throw(td1.deactivate_obs(10), std::invalid_argument);
    td1.deactivate_obs(del_iobs);

    for (int i1 = 0; i1 < td1.R.rows(); i1++) {
        int i2 = i1 + (i1 >= del_iobs ? 1 : 0);
        for (int j1 = 0; j1 < td1.R.cols(); j1++) {
            int j2 = j1 + (j1 >= del_iobs ? 1 : 0);

            test_assert_double_equal(td1.R(i1, j1), td2.R(i2, j2));
        }
        test_assert_double_equal(td1.dObs(i1, 0), td2.dObs(i2, 0));
        test_assert_double_equal(td1.dObs(i1, 1), td2.dObs(i2, 1));

        for (int j1 = 0; j1 < td1.S.cols(); j1++) {
            int j2 = j1;
            test_assert_double_equal(td1.S(i1, j1), td2.S(i2, j2));

            test_assert_double_equal(td1.E(i1, j1), td2.E(i2, j2));

            test_assert_double_equal(td1.D(i1, j1), td2.D(i2, j2));
        }
    }

    td1.deactivate_realization(del_iens);

    for (int i1 = 0; i1 < td1.S.rows(); i1++) {
        int i2 = i1 + (i1 >= del_iobs ? 1 : 0);

        for (int j1 = 0; j1 < td1.S.cols(); j1++) {
            int j2 = j1 + (j1 >= del_iens ? 1 : 0);

            test_assert_double_equal(td1.S(i1, j1), td2.S(i2, j2));

            test_assert_double_equal(td1.E(i1, j1), td2.E(i2, j2));

            test_assert_double_equal(td1.D(i1, j1), td2.D(i2, j2));
        }
    }

    test_assert_int_equal(td1.active_ens_size, ens_size - 1);
    test_assert_int_equal(td1.active_obs_size, obs_size - 1);
}

void test_basic() {
    ecl::util::TestArea work_area("es_testdata");
    int ens_size = 10;
    int obs_size = 7;
    res::es_testdata td1 = make_testdata(ens_size, obs_size);
    td1.save("path/sub/path");
    res::es_testdata td2("path/sub/path");
    test_assert_true(td1.S == td2.S);

    test_assert_int_equal(td1.obs_mask.size(), obs_size);
    test_assert_int_equal(
        std::count(td1.obs_mask.begin(), td1.obs_mask.end(), true), obs_size);

    test_assert_int_equal(td1.ens_mask.size(), ens_size);
    test_assert_int_equal(
        std::count(td1.ens_mask.begin(), td1.ens_mask.end(), true), ens_size);
}

void test_size_problems() {
    ecl::util::TestArea work_area("es_testdata");
    int ens_size = 10;
    int obs_size = 7;
    {
        res::es_testdata td1 = make_testdata(ens_size, obs_size);
        td1.save("path");
    }
    unlink("path/size");
    test_assert_throw(res::es_testdata("path"), std::invalid_argument);
    {
        FILE *fp = util_fopen("path/size", "w");
        fprintf(fp, "%d\n", ens_size);
        fclose(fp);
    }
    test_assert_throw(res::es_testdata("path"), std::invalid_argument);
}

void test_load_state() {
    ecl::util::TestArea work_area("es_testdata");
    int ens_size = 10;
    int obs_size = 7;
    res::es_testdata td0 = make_testdata(ens_size, obs_size);
    td0.save("PATH");

    res::es_testdata td("PATH");
    td.deactivate_realization(5);
    int active_ens_size = td.active_ens_size;

    test_assert_throw(td.make_state("DOES_NOT_EXIST"), std::invalid_argument);

    {
        int invalid_size = 10 * active_ens_size + active_ens_size / 2;
        FILE *stream = util_fopen("PATH/A0", "w");
        for (int i = 0; i < invalid_size; i++)
            fprintf(stream, "%d\n", i);
        fclose(stream);
        test_assert_throw(td.make_state("A0"), std::invalid_argument);
    }
    {
        int state_size = 7;
        int valid_size = state_size * active_ens_size;
        FILE *stream = util_fopen("PATH/A1", "w");
        double value = 0;
        for (int row = 0; row < state_size; row++) {
            for (int iens = 0; iens < active_ens_size; iens++) {
                fprintf(stream, "%lg ", value);
                value++;
            }
            fputs("\n", stream);
        }
        fclose(stream);

        Eigen::MatrixXd A1 = td.make_state("A1");
        test_assert_int_equal(A1.rows(), state_size);
        test_assert_int_equal(A1.cols(), active_ens_size);

        value = 0;
        for (int row = 0; row < state_size; row++) {
            for (int iens = 0; iens < active_ens_size; iens++) {
                test_assert_double_equal(A1(row, iens), value);
                value++;
            }
        }

        td.save_matrix("A2", A1);
        Eigen::MatrixXd A2 = td.make_matrix("A2", state_size, active_ens_size);
        test_assert_true(A1 == A2);
    }
}

int main() {
    test_basic();
    test_load_state();
    test_deactivate();
    test_size_problems();
}
