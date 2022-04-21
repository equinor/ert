/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'local_obsdata_node.h'

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
#ifndef ERT_LOCAL_OBSDATA_NODE_H
#define ERT_LOCAL_OBSDATA_NODE_H

#include <ert/enkf/active_list.hpp>
#include <string>

class LocalObsDataNode {
public:
    LocalObsDataNode(const std::string &key);

    ActiveList *active_list();
    const ActiveList *active_list() const;
    void update_active_list(const ActiveList &al);
    const std::string &name() const;
    bool operator==(const LocalObsDataNode &other) const;
    bool operator!=(const LocalObsDataNode &other) const;

private:
    ActiveList m_active_list;
    std::string m_key;
};

#endif
