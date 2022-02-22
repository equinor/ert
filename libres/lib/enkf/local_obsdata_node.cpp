/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'local_obsdata_node.c'

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
#include <vector>
#include <algorithm>

#include <ert/util/type_macros.h>
#include <ert/util/int_vector.h>
#include <ert/python.hpp>
#include <ert/enkf/local_obsdata_node.hpp>

#define LOCAL_OBSDATA_NODE_TYPE_ID 84441309

struct local_obsdata_node_struct {
    UTIL_TYPE_ID_DECLARATION;
    char *obs_key;
    ActiveList *active_list;
};

UTIL_IS_INSTANCE_FUNCTION(local_obsdata_node, LOCAL_OBSDATA_NODE_TYPE_ID)
UTIL_SAFE_CAST_FUNCTION(local_obsdata_node, LOCAL_OBSDATA_NODE_TYPE_ID)

static local_obsdata_node_type *
local_obsdata_node_alloc__(const char *obs_key) {
    auto node = new local_obsdata_node_type;
    UTIL_TYPE_ID_INIT(node, LOCAL_OBSDATA_NODE_TYPE_ID);
    node->obs_key = util_alloc_string_copy(obs_key);
    node->active_list = nullptr;

    return node;
}

local_obsdata_node_type *local_obsdata_node_alloc(const char *obs_key) {
    local_obsdata_node_type *node = local_obsdata_node_alloc__(obs_key);

    node->active_list = new ActiveList();

    return node;
}

local_obsdata_node_type *
local_obsdata_node_alloc_copy(const local_obsdata_node_type *src) {
    local_obsdata_node_type *target = local_obsdata_node_alloc__(src->obs_key);

    target->active_list = new ActiveList(*src->active_list);

    return target;
}

const char *local_obsdata_node_get_key(const local_obsdata_node_type *node) {
    return node->obs_key;
}

void local_obsdata_node_free(local_obsdata_node_type *node) {
    if (node->active_list)
        delete node->active_list;

    free(node->obs_key);
    delete node;
}

void local_obsdata_node_free__(void *arg) {
    local_obsdata_node_type *node = local_obsdata_node_safe_cast(arg);
    local_obsdata_node_free(node);
}

ActiveList *
local_obsdata_node_get_active_list(const local_obsdata_node_type *node) {
    return node->active_list;
}

ActiveList *
local_obsdata_node_get_copy_active_list(const local_obsdata_node_type *node) {
    return new ActiveList(*node->active_list);
}

namespace {
ActiveList &get_active_list(py::handle obj) {
    auto *obsdata_node = reinterpret_cast<local_obsdata_node_type *>(
        PyLong_AsVoidPtr(obj.attr("_BaseCClass__c_pointer").ptr()));
    auto *active_list = local_obsdata_node_get_active_list(obsdata_node);
    return *active_list;
}
} // namespace

RES_LIB_SUBMODULE("local.local_obsdata_node", m) {
    using namespace py::literals;

    m.def("get_active_list", &get_active_list, "self"_a,
          py::return_value_policy::reference_internal);
}
