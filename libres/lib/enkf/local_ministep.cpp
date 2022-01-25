/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'local_ministep.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include "ert/enkf/row_scaling.hpp"
#include "ert/python.hpp"
#include <stdexcept>
#include <stdlib.h>
#include <string.h>

#include <ert/enkf/local_config.hpp>
#include <ert/enkf/local_ministep.hpp>

/*
   This file implements a 'ministep' configuration for active /
   inactive observations and parameters for ONE enkf update. Observe
   that the updating at one report step can consist of several
   socalled ministeps, i.e. first the northern part of the field with
   the relevant observations, and then the southern part.

   The implementation, in local_ministep_type, is quite simple, it
   only contains the keys for the observations and nodes, with an
   accompanying pointer to an active_list instance which denotes the
   active indices. Observe that this implementation offers no access
   to the internals of the underlying enkf_node / obs_node objects.
*/

UTIL_SAFE_CAST_FUNCTION(local_ministep, LOCAL_MINISTEP_TYPE_ID);
UTIL_IS_INSTANCE_FUNCTION(local_ministep, LOCAL_MINISTEP_TYPE_ID);

local_ministep_type *
local_ministep_alloc(const char *name, analysis_module_type *analysis_module) {
    return new local_ministep_type(name, analysis_module);
}

void local_ministep_free(local_ministep_type *ministep) { delete ministep; }

void local_ministep_free__(void *arg) {
    local_ministep_type *ministep = local_ministep_safe_cast(arg);
    local_ministep_free(ministep);
}

/*
   When adding observations and update nodes here observe the following:

   1. The thing will fail hard if you try to add a node/obs which is
   already in the hash table.

   2. The newly added elements will be assigned an active_list
   instance with mode ALL_ACTIVE.
*/

void local_ministep_add_obsdata(local_ministep_type *ministep,
                                local_obsdata_type *obsdata) {
    if (ministep->observations == NULL)
        ministep->observations = obsdata;
    else { // Add nodes from input observations to existing observations
        int iobs;
        for (iobs = 0; iobs < local_obsdata_get_size(obsdata); iobs++) {
            local_obsdata_node_type *obs_node =
                local_obsdata_iget(obsdata, iobs);
            local_obsdata_node_type *new_node =
                local_obsdata_node_alloc_copy(obs_node);
            local_ministep_add_obsdata_node(ministep, new_node);
        }
    }
}

void local_ministep_add_obs_data(local_ministep_type *ministep,
                                 obs_data_type *obs_data) {
    if (ministep->obs_data != NULL) {
        obs_data_free(ministep->obs_data);
        ministep->obs_data = NULL;
    }
    ministep->obs_data = obs_data;
}

void local_ministep_add_obsdata_node(local_ministep_type *ministep,
                                     local_obsdata_node_type *obsdatanode) {
    local_obsdata_type *obsdata = local_ministep_get_obsdata(ministep);
    local_obsdata_add_node(obsdata, obsdatanode);
}

int local_ministep_num_active_data(const local_ministep_type *ministep) {
    return ministep->num_active_data();
}
active_list_type *
local_ministep_get_active_data_list(const local_ministep_type *ministep,
                                    const char *key) {
    return ministep->get_active_data_list(key);
}
bool local_ministep_data_is_active(const local_ministep_type *ministep,
                                   const char *key) {
    return ministep->data_is_active(key);
}
void local_ministep_activate_data(local_ministep_type *ministep,
                                  const char *key) {
    ministep->add_active_data(key);
}

RowScaling *
local_ministep_get_or_create_row_scaling(local_ministep_type *ministep,
                                         const char *key) {
    auto scaling_iter = ministep->scaling.find(key);
    if (scaling_iter == ministep->scaling.end()) {
        if (!hash_has_key(ministep->active_size, key))
            throw std::invalid_argument(
                "Tried to create row_scaling object for unknown key");

        ministep->scaling.emplace(key, std::make_shared<RowScaling>());
    }
    return ministep->scaling[key].get();
}

local_obsdata_type *
local_ministep_get_obsdata(const local_ministep_type *ministep) {
    return ministep->observations;
}

obs_data_type *
local_ministep_get_obs_data(const local_ministep_type *ministep) {
    return ministep->obs_data;
}

const char *local_ministep_get_name(const local_ministep_type *ministep) {
    return ministep->name.data();
}

bool local_ministep_has_analysis_module(const local_ministep_type *ministep) {
    return ministep->analysis_module != NULL;
}

analysis_module_type *
local_ministep_get_analysis_module(const local_ministep_type *ministep) {
    return ministep->analysis_module;
}

void local_ministep_summary_fprintf(const local_ministep_type *ministep,
                                    FILE *stream) {

    fprintf(stream, "MINISTEP:%s,", ministep->name.data());

    {
        hash_iter_type *data_iter = hash_iter_alloc(ministep->active_size);
        while (!hash_iter_is_complete(data_iter)) {
            const char *data_key = hash_iter_get_next_key(data_iter);
            fprintf(stream, "NAME OF DATA:%s,", data_key);

            active_list_type *active_list =
                (active_list_type *)hash_get(ministep->active_size, data_key);
            active_list_summary_fprintf(active_list, ministep->name.data(),
                                        data_key, stream);
        }
        hash_iter_free(data_iter);

        /* Only one OBSDATA */
        local_obsdata_type *obsdata = local_ministep_get_obsdata(ministep);
        local_obsdata_summary_fprintf(obsdata, stream);
        fprintf(stream, "\n");
    }
}

namespace {
RowScaling *get_or_create_row_scaling(py::handle obj, const std::string &name) {
    auto ministep = reinterpret_cast<local_ministep_type *>(
        PyLong_AsVoidPtr(obj.attr("_BaseCClass__c_pointer").ptr()));
    auto row_scaling =
        local_ministep_get_or_create_row_scaling(ministep, name.c_str());
    return row_scaling;
}
} // namespace

RES_LIB_SUBMODULE("local.ministep", m) {
    using namespace py::literals;

    m.def("get_or_create_row_scaling", &get_or_create_row_scaling, "self"_a,
          "name"_a);
}
