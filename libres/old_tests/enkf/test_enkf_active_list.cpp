/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_active_list.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/enkf/active_list.hpp>

int main(int argc, char **argv) {
    ActiveList active_list1;
    ActiveList active_list2;

    test_assert_true(active_list1 == active_list2);

    active_list1.add_index(11);
    test_assert_false(active_list1 == active_list2);

    active_list1.add_index(12);
    test_assert_false(active_list1 == active_list2);

    active_list2.add_index(11);
    test_assert_false(active_list1 == active_list2);

    active_list2.add_index(12);
    test_assert_true(active_list1 == active_list2);

    active_list2.add_index(11);
    test_assert_true(active_list1 == active_list2);

    active_list2.add_index(13);
    test_assert_false(active_list1 == active_list2);

    active_list1.add_index(13);
    test_assert_true(active_list1 == active_list2);
    exit(0);
}
