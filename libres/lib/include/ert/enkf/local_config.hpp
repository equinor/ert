/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'local_config.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_LOCAL_CONFIG_H
#define ERT_LOCAL_CONFIG_H

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ert/enkf/local_updatestep.hpp>
#include <ert/enkf/local_ministep.hpp>
#include <ert/enkf/local_obsdata.hpp>

class LocalConfig {
public:
    LocalConfig(const std::vector<std::string> &parameter_keys,
                const std::vector<std::string> &obs_keys);

    bool has_obsdata(const std::string &obs_key) const;

    LocalMinistep &make_ministep(const std::string &key);
    LocalMinistep *ministep(const std::string &key);
    LocalMinistep *global_ministep();

    LocalObsData *make_obsdata(const std::string &key);
    LocalObsData *obsdata(const std::string &key);
    LocalObsData *global_obsdata();
    LocalUpdateStep &updatestep();

    void clear() const;

private:
    LocalMinistep m_global_ministep;
    LocalObsData m_global_obsdata;
    LocalUpdateStep m_global_updatestep;

    std::optional<LocalUpdateStep> m_updatestep;
    std::unordered_map<std::string, LocalMinistep> m_ministep;
    std::unordered_map<std::string, LocalObsData> m_obsdata;
};

#endif
