/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'local_obsdata.h'

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
#ifndef ERT_LOCAL_OBSDATA_H
#define ERT_LOCAL_OBSDATA_H

#include <string>
#include <unordered_map>
#include <vector>

#include <ert/enkf/local_obsdata_node.hpp>

class LocalObsData {
public:
    explicit LocalObsData(const std::string &name);
    static LocalObsData make_wrapper(const LocalObsDataNode &node);
    const LocalObsDataNode &operator[](const std::string &name) const;
    const LocalObsDataNode &operator[](std::size_t index) const;
    std::vector<LocalObsDataNode>::const_iterator begin() const;
    std::vector<LocalObsDataNode>::const_iterator end() const;
    void del_node(const std::string &key);
    bool has_node(const std::string &key) const;
    bool add_node(const std::string &key);
    bool add_node(const LocalObsDataNode &node);
    std::size_t size() const;
    bool empty() const;
    const std::string &name() const;

    bool operator==(const LocalObsData &other) const;

private:
    std::vector<LocalObsDataNode> m_nodes;
    std::unordered_map<std::string, std::size_t> m_node_index;
    std::string m_name;
};

#endif
