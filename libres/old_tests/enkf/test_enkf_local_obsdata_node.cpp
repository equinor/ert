/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_local_obsdata_node.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/test_util.h>

#include <ert/enkf/local_obsdata_node.hpp>

void test_content(LocalObsDataNode *node) {
    const auto *active_list = node->active_list();

    test_assert_not_NULL(active_list);

    {
        ActiveList new_active_list;

        new_active_list.add_index(1098);
        test_assert_false(new_active_list == *active_list);
    }
}

int main(int argc, char **argv) {
    const char *obs_key = "1234";
    LocalObsDataNode node(obs_key);

    test_assert_string_equal(obs_key, node.name().c_str());
    test_content(&node);
    exit(0);
}
