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

#include <ert/enkf/local_obsdata.hpp>

#include "ert/python.hpp"

LocalObsData::LocalObsData(const std::string &name) : m_name(name) {}

LocalObsData LocalObsData::make_wrapper(const LocalObsDataNode &node) {
    LocalObsData obs_data(node.name());
    obs_data.add_node(node);
    return obs_data;
}

const LocalObsDataNode &
LocalObsData::operator[](const std::string &obs_key) const {
    auto index = this->m_node_index.at(obs_key);
    return this->m_nodes[index];
}

const LocalObsDataNode &LocalObsData::operator[](std::size_t index) const {
    return this->m_nodes[index];
}

void LocalObsData::del_node(const std::string &key) {
    auto index = this->m_node_index.at(key);
    this->m_nodes.erase(this->m_nodes.begin() + index);
    this->m_node_index.clear();
    for (index = 0; index < this->m_nodes.size(); index++) {
        const auto &node = this->m_nodes[index];
        this->m_node_index.insert({node.name(), index});
    }
}

bool LocalObsData::has_node(const std::string &key) {
    return (this->m_node_index.count(key) != 0);
}

bool LocalObsData::add_node(const std::string &key) {
    if (this->m_node_index.count(key) > 0)
        throw std::out_of_range("Key already registered");

    this->m_nodes.emplace_back(key);
    this->m_node_index.emplace(key, this->m_nodes.size() - 1);
    return true;
}

bool LocalObsData::add_node(const LocalObsDataNode &node) {
    if (this->has_node(node.name()))
        return false;

    this->m_nodes.push_back(node);
    this->m_node_index.emplace(node.name(), this->m_nodes.size() - 1);
    return true;
}

std::size_t LocalObsData::size() const { return this->m_nodes.size(); }

const std::string &LocalObsData::name() const { return this->m_name; }

std::vector<LocalObsDataNode>::const_iterator LocalObsData::begin() const {
    return this->m_nodes.begin();
}

std::vector<LocalObsDataNode>::const_iterator LocalObsData::end() const {
    return this->m_nodes.end();
}

bool LocalObsData::operator==(const LocalObsData& other) const {
    return this->m_nodes == other.m_nodes &&
           this->m_node_index == other.m_node_index &&
           this->m_name == other.m_name;
}

RES_LIB_SUBMODULE("local.local_obsdata", m) {
    py::class_<LocalObsData>(m, "LocalObsdata")
        .def(py::init<const std::string &>())
        .def("__len__", &LocalObsData::size)
        .def("__getitem__", py::overload_cast<std::size_t>(
                                &LocalObsData::operator[], py::const_))
        .def("__getitem__", py::overload_cast<const std::string &>(
                                &LocalObsData::operator[], py::const_))
        .def("__contains__", &LocalObsData::has_node)
        .def("__delitem__", &LocalObsData::del_node)
        .def("addNode",
             py::overload_cast<const std::string &>(&LocalObsData::add_node))
        .def("name", &LocalObsData::name)
        .def("getActiveList",
             [](const LocalObsData &obs_data, const std::string &key) {
                 const auto &node = obs_data[key];
                 return node.active_list();
             });
}
