/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'local_obsdata.h'

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
#ifndef ERT_LOCAL_OBSDATA_H
#define ERT_LOCAL_OBSDATA_H

#include <stdbool.h>
#include <vector>

#include <ert/util/type_macros.h>

#include <ert/enkf/local_obsdata_node.hpp>

typedef struct local_obsdata_struct local_obsdata_type;

void local_obsdata_free__(void *arg);
extern "C" bool local_obsdata_has_node(const local_obsdata_type *data,
                                       const char *key);
local_obsdata_type *local_obsdata_alloc_copy(const local_obsdata_type *src,
                                             const char *target_key);
extern "C" local_obsdata_type *local_obsdata_alloc(const char *name);
extern "C" void local_obsdata_free(local_obsdata_type *data);
extern "C" int local_obsdata_get_size(const local_obsdata_type *data);
extern "C" bool local_obsdata_add_node(local_obsdata_type *data,
                                       const LocalObsDataNode *node);
extern "C" const LocalObsDataNode *
local_obsdata_iget(const local_obsdata_type *data, int index);
local_obsdata_type *local_obsdata_alloc_wrapper(LocalObsDataNode *node);
extern "C" const char *local_obsdata_get_name(const local_obsdata_type *data);
extern "C" LocalObsDataNode *local_obsdata_get(const local_obsdata_type *data,
                                               const char *key);
extern "C" void local_obsdata_del_node(local_obsdata_type *data,
                                       const char *key);
extern "C" ActiveList *
local_obsdata_get_node_active_list(const local_obsdata_type *obsdata,
                                   const char *obs_key);
void local_obsdata_summary_fprintf(const local_obsdata_type *obsdata,
                                   FILE *stream);
extern "C" PY_USED ActiveList *
local_obsdata_get_copy_node_active_list(const local_obsdata_type *obsdata,
                                        const char *obs_key);

UTIL_IS_INSTANCE_HEADER(local_obsdata);

#endif
