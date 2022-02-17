/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'local_updatestep.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_LOCAL_UPDATESTEP_H
#define ERT_LOCAL_UPDATESTEP_H

#include <functional>
#include <vector>

#include <ert/enkf/local_ministep.hpp>
#include <ert/enkf/local_obsdata.hpp>

class LocalUpdateStep {
public:
    explicit LocalUpdateStep(const std::string &name);
    std::size_t size() const;
    const std::string &name() const;
    LocalMinistep *operator[](std::size_t index);
    void add_ministep(LocalMinistep *);

private:
    std::string m_name;
    std::vector<std::reference_wrapper<LocalMinistep>> m_ministep;
};

#endif
