/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'active_list.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <algorithm>
#include <stdexcept>

#include <ert/enkf/active_list.hpp>
#include "ert/python.hpp"

/*
   This file implements a small structure used to denote which
   elements of a node/observation which is active. At the lowest level
   the active elements in a node is just a list of integers. This
   list of integers, with som extra twists is what is implemented
   here.

   All the xxx_config objects have a pointer to an active_list
   instance. This pointer is passed to the enkf_serialize /
   enkf_deserialize routines.

   Observe that for the (very important!!) special case that all
   elements are active the (int *) pointer should not be accessed, and
   the code here is free to return NULL.


Example
-------

Consider a situation where faults number 0,4 and 5 should be active in
a fault object. Then the code will be like:


   ....
   active_list_add_index(multflt_config->active_list , 0);
   active_list_add_index(multflt_config->active_list , 4);
   active_list_add_index(multflt_config->active_list , 5);
   ....

   When this fault object is serialized/deserialized only the elements
   0,4,5 are updated.
*/

void ActiveList::add_index(int new_index) {
    auto iter = std::find(this->m_index_list.begin(), this->m_index_list.end(),
                          new_index);
    printf("ptr: %p trying to insert %d  mode: %d -> ", this, new_index, static_cast<int>(this->m_mode));
    if (iter == this->m_index_list.end()) {
        this->m_index_list.push_back(new_index);
        this->m_mode = PARTLY_ACTIVE;
    }
    printf("%d \n", static_cast<int>(this->m_mode));
}

/*
   When mode == PARTLY_ACTIVE the active_list instance knows the size
   of the active set; if the mode is INACTIVE 0 will be returned and
   if the mode is ALL_ACTIVE the input parameter @total_size will be
   passed back to calling scope.
*/

int ActiveList::getActiveSize(int total_size) const {
    int active_size;
    switch (this->m_mode) {
    case PARTLY_ACTIVE:
        return this->m_index_list.size();
    case ALL_ACTIVE:
        return total_size;
    default:
        throw std::logic_error("Unhandled enum value");
    }
}

active_mode_type ActiveList::getMode() const { return this->m_mode; }

/*
   This will return a (const int *) pointer to the active indices. IFF
   (mode == INACTIVE || mode == ALL_ACTIVE) it will instead just
   return NULL. In that case it is the responsability of the calling
   scope to not dereference the NULL pointer.
*/

const int *ActiveList::active_list_get_active() const {
    if (this->m_mode == PARTLY_ACTIVE)
        return this->m_index_list.data();
    else
        return nullptr;
}

const std::vector<int> &ActiveList::index_list() const {
    return this->m_index_list;
}

void ActiveList::summary_fprintf(const char *dataset_key, const char *key,
                                 FILE *stream) const {
    int number_of_active = this->m_index_list.size();
    if (this->m_mode == ALL_ACTIVE) {
        fprintf(stream, "NUMBER OF ACTIVE:%d,STATUS:%s,", number_of_active,
                "ALL_ACTIVE");
    } else if (this->m_mode == PARTLY_ACTIVE) {
        fprintf(stream, "NUMBER OF ACTIVE:%d,STATUS:%s,", number_of_active,
                "PARTLY_ACTIVE");
    } else
        fprintf(stream, "NUMBER OF ACTIVE:%d,STATUS:%s,", number_of_active,
                "INACTIVE");
}

bool ActiveList::operator==(const ActiveList &other) const {
    if (this == &other)
        return true;

    if (this->m_mode != other.m_mode)
        return false;

    if (this->m_mode == PARTLY_ACTIVE)
        return this->m_index_list == other.m_index_list;

    return true;
}

namespace {

std::string repr(const ActiveList& al) {
    if (al.getMode() == PARTLY_ACTIVE)
        return fmt::format("ActiveList(mode=PARTLY_ACTIVE, active_size = {})", al.getActiveSize(0));

    return "ActiveList(mode=ALL_ACTIVE)";
}

}

RES_LIB_SUBMODULE("local.active_list", m) {
    py::class_<ActiveList>(m, "ActiveList")
        .def(py::init<>())
        .def("getMode", &ActiveList::getMode)
        .def("get_active_index_list", &ActiveList::index_list)
        .def("addActiveIndex", &ActiveList::add_index)
        .def("__repr__", &repr)
        .def("print_self", &ActiveList::print_self)
        .def("getActiveSize", &ActiveList::getActiveSize);

    py::enum_<active_mode_type>(m, "ActiveMode", py::arithmetic())
        .value("ALL_ACTIVE", ALL_ACTIVE)
        .value("PARTLY_ACTIVE", PARTLY_ACTIVE)
        .export_values();
}
