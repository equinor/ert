/*
   Copyright (C) 2020  Equinor ASA, Norway.

   The file 'enkf_local_dataset.cpp' is part of ERT - Ensemble based Reservoir Tool.

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

#include <stdlib.h>
#include <stdexcept>
#include <ert/util/test_util.hpp>
#include <ert/enkf/local_dataset.hpp>


void test_create() {
  local_dataset_type * ld = local_dataset_alloc("DATA");
  local_dataset_add_node(ld, "PERMX");
  local_dataset_add_node(ld, "PERMY");
  local_dataset_add_node(ld, "PERMZ");
  test_assert_false( local_dataset_has_row_scaling(ld, "PERMX"));
  test_assert_false( local_dataset_has_row_scaling(ld, "PERMY"));
  test_assert_false( local_dataset_has_row_scaling(ld, "PERMZ"));

  {
    stringlist_type * unscaled_keys = local_dataset_alloc_unscaled_keys(ld);
    test_assert_int_equal( stringlist_get_size(unscaled_keys), 3 );
    test_assert_true( stringlist_contains(unscaled_keys, "PERMX"));
    test_assert_true( stringlist_contains(unscaled_keys, "PERMY"));
    test_assert_true( stringlist_contains(unscaled_keys, "PERMZ"));
    stringlist_free(unscaled_keys);
  }
  {
    stringlist_type * scaled_keys = local_dataset_alloc_scaled_keys(ld);
    test_assert_int_equal( stringlist_get_size(scaled_keys), 0 );
    stringlist_free(scaled_keys);
  }

  local_dataset_free(ld);
}

int main(int argc , char ** argv) {
  test_create();
}

