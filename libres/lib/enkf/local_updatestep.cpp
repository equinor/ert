/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'local_updatestep.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <algorithm>

#include <ert/enkf/local_ministep.hpp>
#include <ert/enkf/local_updatestep.hpp>

#include "ert/python.hpp"

/*
   One enkf update is described/configured by the data structure in
   local_ministep.c. This file implements a local report_step, which
   is a collection of ministeps - in many cases a local_updatestep will
   only consist of one single local_ministep; but in principle it can
   contain several.
*/

LocalUpdateStep::LocalUpdateStep(const std::string &name) : m_name(name) {}

std::size_t LocalUpdateStep::size() const { return this->m_ministep.size(); }

const std::string &LocalUpdateStep::name() const { return this->m_name; }

LocalMinistep &LocalUpdateStep::operator[](std::size_t index) {
    return this->m_ministep[index].get();
}

const LocalMinistep &LocalUpdateStep::operator[](std::size_t index) const {
    return this->m_ministep[index].get();
}

void LocalUpdateStep::add_ministep(LocalMinistep &ministep) {
    this->m_ministep.push_back(std::ref(ministep));
}

RES_LIB_SUBMODULE("local.updatestep", m) {

    auto get_ministep =
        static_cast<LocalMinistep &(LocalUpdateStep::*)(std::size_t index)>(
            &LocalUpdateStep::operator[]);

    py::class_<LocalUpdateStep>(m, "LocalUpdateStep")
        .def(py::init<const std::string &>())
        .def("__len__", &LocalUpdateStep::size)
        .def("name", &LocalUpdateStep::name)
        .def("__getitem__", get_ministep,
             py::return_value_policy::reference_internal)
        .def("attachMinistep", &LocalUpdateStep::add_ministep);
}
