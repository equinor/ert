/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'local_obsdata.c'

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
#include <unordered_map>
#include <vector>

#include <ert/util/util.h>
#include <ert/util/type_macros.h>
#include <ert/util/vector.h>
#include <ert/util/hash.h>

#include <ert/python.hpp>
#include <ert/enkf/local_config.hpp>

#define LOCAL_OBSDATA_TYPE_ID 86331309

struct local_obsdata_struct {
    UTIL_TYPE_ID_DECLARATION;
    std::unordered_map<std::string, std::size_t> node_index;
    std::vector<LocalObsDataNode> nodes;
    char *name;
};

UTIL_IS_INSTANCE_FUNCTION(local_obsdata, LOCAL_OBSDATA_TYPE_ID)
static UTIL_SAFE_CAST_FUNCTION(local_obsdata, LOCAL_OBSDATA_TYPE_ID)

    local_obsdata_type *local_obsdata_alloc(const char *name) {
    local_obsdata_type *data = new local_obsdata_type();
    UTIL_TYPE_ID_INIT(data, LOCAL_OBSDATA_TYPE_ID);
    data->name = util_alloc_string_copy(name);
    return data;
}

local_obsdata_type *local_obsdata_alloc_copy(const local_obsdata_type *src,
                                             const char *target_key) {
    local_obsdata_type *target = local_obsdata_alloc(target_key);
    for (int i = 0; i < local_obsdata_get_size(src); i++) {
        const auto *node = local_obsdata_iget(src, i);
        local_obsdata_add_node(target, node);
    }
    return target;
}

void local_obsdata_free(local_obsdata_type *data) {
    free(data->name);
    delete data;
}

void local_obsdata_free__(void *arg) {
    local_obsdata_type *data = local_obsdata_safe_cast(arg);
    return local_obsdata_free(data);
}

const char *local_obsdata_get_name(const local_obsdata_type *data) {
    return data->name;
}

int local_obsdata_get_size(const local_obsdata_type *data) {
    return data->nodes.size();
}

/*
  The @data instance will insert a copy
*/

bool local_obsdata_add_node(local_obsdata_type *data,
                            const LocalObsDataNode *node) {
    const char *key = node->name().c_str();
    if (local_obsdata_has_node(data, key))
        return false;
    else {
        data->node_index.emplace(key, data->nodes.size());
        data->nodes.push_back(*node);
        return true;
    }
}

void local_obsdata_del_node(local_obsdata_type *data, const char *key) {
    auto index = data->node_index.at(key);
    data->nodes.erase(data->nodes.begin() + index);
    data->node_index.erase(key);
}

const LocalObsDataNode *local_obsdata_iget(const local_obsdata_type *data,
                                           int index) {
    return &data->nodes[index];
}

LocalObsDataNode *local_obsdata_get(local_obsdata_type *data, const char *key) {
    auto index = data->node_index.at(key);
    return &data->nodes[index];
}

bool local_obsdata_has_node(const local_obsdata_type *data, const char *key) {
    return data->node_index.count(key) == 1;
}

const ActiveList *
local_obsdata_get_node_active_list(local_obsdata_type *obsdata,
                                   const char *key) {
    auto index = obsdata->node_index.at(key);
    auto &node = obsdata->nodes[index];
    return node.active_list();
}

namespace {
const ActiveList *get_active_list(py::handle obj, const std::string &name) {
    auto *self = ert::from_cwrap<local_obsdata_type>(obj);
    auto active_list = local_obsdata_get_node_active_list(self, name.c_str());
    return active_list;
}

bool add_node(py::handle obj, const LocalObsDataNode &node) {
    auto *obsdata = ert::from_cwrap<local_obsdata_type>(obj);
    return local_obsdata_add_node(obsdata, &node);
}

const LocalObsDataNode &get_node(py::handle obj, const std::string &key) {
    auto *obsdata = ert::from_cwrap<local_obsdata_type>(obj);
    auto *node_ptr = local_obsdata_get(obsdata, key.c_str());
    return *node_ptr;
}

const LocalObsDataNode &iget_node(py::handle obj, int index) {
    auto *obsdata = ert::from_cwrap<local_obsdata_type>(obj);
    auto *node_ptr = local_obsdata_iget(obsdata, index);
    return *node_ptr;
}

} // namespace

RES_LIB_SUBMODULE("local.local_obsdata", m) {
    using namespace py::literals;

    m.def("get_active_list", &get_active_list, "self"_a, "key"_a,
          py::return_value_policy::reference_internal);
    m.def("copy_active_list", &get_active_list, "self"_a, "key"_a,
          py::return_value_policy::copy);
    m.def("add_node", &add_node, "self"_a, "node"_a);
    m.def("iget_node", &iget_node, "self"_a, "index"_a,
          py::return_value_policy::reference_internal);
    m.def("get_node", &get_node, "self"_a, "key"_a,
          py::return_value_policy::reference_internal);
}
