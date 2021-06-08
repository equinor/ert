/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'job_kw_definitions.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_JOB_KW_DEFINITIONS_H
#define ERT_JOB_KW_DEFINITIONS_H
#ifdef __cplusplus
extern "C" {
#endif

#define MIN_ARG_KEY    "MIN_ARG"
#define MAX_ARG_KEY    "MAX_ARG"
#define ARG_TYPE_KEY   "ARG_TYPE"
#define EXECUTABLE_KEY "EXECUTABLE"

#define JOB_STRING_TYPE       "STRING"
#define JOB_INT_TYPE          "INT"
#define JOB_FLOAT_TYPE        "FLOAT"
#define JOB_BOOL_TYPE         "BOOL"
#define JOB_RUNTIME_FILE_TYPE "RUNTIME_FILE"
#define JOB_RUNTIME_INT_TYPE  "RUNTIME_INT"

config_item_types job_kw_get_type(const char * arg_type);


#ifdef __cplusplus
}
#endif
#endif
