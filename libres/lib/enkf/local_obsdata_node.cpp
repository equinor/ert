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
#include <ert/enkf/local_obsdata_node.hpp>

#include "ert/python.hpp"

RES_LIB_SUBMODULE("local.local_obsdata_node", m) {
    py::class_<LocalObsDataNode>(m, "LocalObsdataNode")
        .def(py::init<const std::string &>())
        .def("key", &LocalObsDataNode::name)
        .def("getKey", &LocalObsDataNode::name)
        .def(pybind11::self == pybind11::self)
        .def(pybind11::self != pybind11::self)
        .def_readwrite("active_list", &LocalObsDataNode::second);
}
