#pragma once

#include <string>
#include <ert/enkf/active_list.hpp>

class LocalObsDataNode : public std::pair<std::string, ActiveList> {
public:
    LocalObsDataNode(const std::string &name) : pair(name, {}) {}

    const std::string &name() const { return first; }
    ActiveList &active_list() { return second; }
    const ActiveList &active_list() const { return second; }
};
