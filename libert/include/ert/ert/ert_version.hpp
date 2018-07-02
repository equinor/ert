/*
   Copyright (C) 2017  Statoil ASA, Norway.

   The file 'ert_version.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_VERSION
#define ERT_VERSION

#include <stdbool.h>

#ifdef __cplusplus
extern"C" {
#endif

const char * ert_version_get_git_commit();
const char * ert_version_get_git_commit_short();
const char * ert_version_get_build_time();
int    	     ert_version_get_major_version();
int    	     ert_version_get_minor_version();
const char * ert_version_get_micro_version();
bool         ert_version_is_devel_version();

#ifdef __cplusplus
}
#endif

#endif
