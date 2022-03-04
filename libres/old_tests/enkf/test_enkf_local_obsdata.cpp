/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_local_obsdata.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/enkf/local_obsdata.hpp>

void test_wrapper() {
    LocalObsDataNode node("KEY");
    auto data = LocalObsData::make_wrapper(node);
    test_assert_int_equal(1, data.size());
    test_assert_true(node == data[0]);
    test_assert_true(data.has_node("KEY"));
    test_assert_false(data.has_node("KEYX"));
    test_assert_string_equal(node.name().c_str(), data.name().c_str());
}

int main(int argc, char **argv) {
    LocalObsData obsdata("KEY");

    test_assert_int_equal(0, obsdata.size());
    test_assert_string_equal("KEY", obsdata.name().c_str());

    {
        LocalObsDataNode obsnode("KEY");
        test_assert_true(obsdata.add_node(obsnode));
        test_assert_false(obsdata.add_node(obsnode));
        test_assert_int_equal(1, obsdata.size());
        test_assert_true(obsnode == obsdata[0]);
    }

    test_wrapper();
    exit(0);
}
