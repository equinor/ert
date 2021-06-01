/*
   Copyright (C) 2016  Equinor ASA, Norway.

   The file 'ecl_version.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ECL_VERSION
#define ECL_VERSION

#include <stdbool.h>
#include <ert/tooling.hpp>

#ifdef __cplusplus
extern"C" {
#endif

PY_USED int res_version_get_major_version();
PY_USED int res_version_get_minor_version();
PY_USED const char * res_version_get_micro_version();
PY_USED const char * res_version_get_git_commit();
PY_USED const char * res_version_get_build_time();
PY_USED bool res_version_is_devel_version();

#ifdef __cplusplus
}
#endif

#endif
