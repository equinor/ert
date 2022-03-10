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
#include <stdexcept>
#include <stdlib.h>
#include <string.h>

#include "ert/python.hpp"

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

RowScaling &LocalMinistep::get_or_create_row_scaling(const std::string &key) {
    auto scaling_iter = this->scaling.find(key);
    if (scaling_iter == this->scaling.end()) {
        if (!this->data_is_active(key))
            throw py::key_error(
                "Tried to create row_scaling object for unknown key");

        this->scaling.emplace(key, RowScaling());
    }
    return this->scaling.at(key);
}

ActiveList &LocalMinistep::get_active_data_list(const std::string &node_key) {
    auto iter = this->active_size.find(node_key);
    if (iter == this->active_size.end())
        throw py::key_error(fmt::format(
            "Tried to get ActiveList from unknown key: {}", node_key));

    return iter->second;
}

const ActiveList &
LocalMinistep::get_active_data_list(const std::string &node_key) const {
    auto iter = this->active_size.find(node_key);
    if (iter == this->active_size.end())
        throw py::key_error(fmt::format(
            "Tried to get ActiveList from unknown key: {}", node_key));

    return iter->second;
}

const RowScaling &LocalMinistep::get_row_scaling(const std::string &key) const {
    auto iter = this->scaling.find(key);
    if (iter == this->scaling.end())
        throw py::key_error(
            fmt::format("No such RowScaling key registered: {}", key));

    return iter->second;
}

RowScaling &LocalMinistep::get_row_scaling(const std::string &key) {
    auto iter = this->scaling.find(key);
    if (iter == this->scaling.end())
        throw py::key_error(
            fmt::format("No such RowScaling key registered: {}", key));

    return iter->second;
}

RES_LIB_SUBMODULE("local.ministep", m) {
    using namespace py::literals;

    auto get_active_data_list =
        static_cast<ActiveList &(LocalMinistep::*)(const std::string &)>(
            &LocalMinistep::get_active_data_list);
    auto get_obs_data = static_cast<LocalObsData &(LocalMinistep::*)()>(
        &LocalMinistep::get_obsdata);
    auto get_row_scaling =
        static_cast<RowScaling &(LocalMinistep::*)(const std::string &)>(
            &LocalMinistep::get_row_scaling);

    py::class_<LocalMinistep>(m, "LocalMinistep")
        .def("hasActiveData", &LocalMinistep::data_is_active)
        .def("getActiveList", get_active_data_list,
             py::return_value_policy::reference_internal)
        .def("numActiveData", &LocalMinistep::num_active_data)
        .def("addActiveData", &LocalMinistep::add_active_data)
        .def("addNode",
             [](LocalMinistep &self, const LocalObsDataNode &node) {
                 auto &observations = self.get_obsdata();
                 observations.add_node(node);
             })
        .def("attachObsset", &LocalMinistep::add_obsdata)
        .def("row_scaling", &LocalMinistep::get_or_create_row_scaling,
             py::return_value_policy::reference_internal)
        .def("getLocalObsData", get_obs_data,
             py::return_value_policy::reference_internal)
        .def("have_obsdata", &LocalMinistep::have_obsdata)
        .def("name", &LocalMinistep::name)
        .def("get_runtime_obs_active_list",
             &LocalMinistep::get_runtime_obs_active_list)
        .def("get_or_create_row_scaling",
             &LocalMinistep::get_or_create_row_scaling, "name"_a,
             py::return_value_policy::reference_internal);
}
