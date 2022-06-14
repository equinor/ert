/*
   Copyright (C) 2012  Equinor ASA, Norway.

   The file 'ert_util_PATH_test.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <string.h>

#include <ert/util/test_util.hpp>
#include <ert/util/vector.hpp>

#include <ert/res_util/res_env.hpp>

int main(int argc, char **argv) {
    setenv("PATH", "/usr/bin:/bin:/usr/local/bin", 1);
    auto path_list = res_env_alloc_PATH_list();
    if (path_list[0].compare("/usr/bin"))
        test_error_exit("Failed on first path element\n");

    if (path_list[1].compare("/bin"))
        test_error_exit("Failed on second path element\n");

    if (path_list[2].compare("/usr/local/bin"))
        test_error_exit("Failed on third  path element\n");

    exit(0);
}
