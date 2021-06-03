/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'trans_func.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/test_util.h>
#include <ert/util/stringlist.h>

#include <ert/enkf/trans_func.hpp>



void test_triangular() {
  stringlist_type * args = stringlist_alloc_new();
  stringlist_append_copy(args , "TRIANGULAR");
  stringlist_append_copy(args, "0");
  stringlist_append_copy(args,"0.5");
  stringlist_append_copy(args, "1.0");

  trans_func_type * trans_func = trans_func_alloc(args);
  test_assert_double_equal( trans_func_eval(trans_func, 0.0), 0.50);
  trans_func_free( trans_func );
  stringlist_free(args);
}

void test_triangular_assymetric() {
  stringlist_type * args = stringlist_alloc_new();
  stringlist_append_copy(args , "TRIANGULAR");
  stringlist_append_copy(args, "0");
  stringlist_append_copy(args,"1.0");
  stringlist_append_copy(args, "4.0");

  trans_func_type * trans_func = trans_func_alloc(args);
  test_assert_double_equal( trans_func_eval(trans_func, -1.0), 0.7966310411513150456286);
  test_assert_double_equal( trans_func_eval(trans_func, 1.1), 2.72407181575270778882286);
  trans_func_free( trans_func );
  stringlist_free(args);
}

void test_create() {
  {
    stringlist_type * args = stringlist_alloc_new();
    stringlist_append_copy(args , "UNKNOWN_FUNCTION");
    test_assert_NULL( trans_func_alloc(args));
    stringlist_free(args);
  }
  {
    stringlist_type * args = stringlist_alloc_new();
    stringlist_append_copy(args , "UNIFORM");
    stringlist_append_copy(args, "0");
    stringlist_append_copy(args,"1");

    trans_func_type * trans_func = trans_func_alloc(args);
    test_assert_double_equal( trans_func_eval(trans_func, 0.0), 0.50);
    trans_func_free( trans_func );

    stringlist_free(args);
  }
  {
    stringlist_type * args = stringlist_alloc_new();
    stringlist_append_copy(args , "UNIFORM");
    stringlist_append_copy(args, "0");
    stringlist_append_copy(args,"X");
    test_assert_NULL( trans_func_alloc(args));
    stringlist_free(args);
  }
  {
    stringlist_type * args = stringlist_alloc_new();
    stringlist_append_copy(args , "UNIFORM");
    stringlist_append_copy(args, "0");
    test_assert_NULL( trans_func_alloc(args));
    stringlist_free(args);
  }
}

int main(int argc , char ** argv) {
  test_create();
  test_triangular();
  test_triangular_assymetric();
}

