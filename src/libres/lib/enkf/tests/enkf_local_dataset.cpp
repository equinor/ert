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

#include <algorithm>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>
#include <ert/util/test_util.hpp>
#include <ert/enkf/local_dataset.hpp>


bool vector_contains(const std::vector<std::string>& keys, const std::string& key) {
  auto iter = std::find(keys.begin(), keys.end(), key);
  return (iter != keys.end());
}


void test_create() {
    local_dataset_type * ld = local_dataset_alloc("DATA");
    local_dataset_add_node(ld, "PERMX");
    local_dataset_add_node(ld, "PERMY");
    local_dataset_add_node(ld, "PERMZ");
    test_assert_false( local_dataset_has_row_scaling(ld, "PERMX"));
    test_assert_false( local_dataset_has_row_scaling(ld, "PERMY"));
    test_assert_false( local_dataset_has_row_scaling(ld, "PERMZ"));

    const auto& unscaled_keys = local_dataset_unscaled_keys(ld);
    test_assert_int_equal( unscaled_keys.size(), 3 );
    test_assert_true( vector_contains(unscaled_keys, "PERMX"));
    test_assert_true( vector_contains(unscaled_keys, "PERMY"));
    test_assert_true( vector_contains(unscaled_keys, "PERMZ"));

    const auto& scaled_keys = local_dataset_scaled_keys(ld);
    test_assert_int_equal( scaled_keys.size(), 0 );

    local_dataset_free(ld);
}


void test_create_row_scaling() {
  local_dataset_type * ld = local_dataset_alloc("DATA");
  test_assert_throw(local_dataset_get_or_create_row_scaling(ld, "NO_SUCH_KEY"), std::invalid_argument);

  local_dataset_add_node(ld, "PERMX");
  row_scaling_type * rs = local_dataset_get_or_create_row_scaling(ld, "PERMX");
  test_assert_true( local_dataset_has_row_scaling(ld, "PERMX"));

  test_assert_false( local_dataset_has_row_scaling(ld, "PERMY"));

  local_dataset_add_node(ld, "PERMZ");
  test_assert_false( local_dataset_has_row_scaling(ld, "PERMZ"));

  local_dataset_free(ld);
}

int main(int argc , char ** argv) {
  test_create();
  test_create_row_scaling();
}

