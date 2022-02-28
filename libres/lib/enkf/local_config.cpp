/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'local_config.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdio.h>

#include <ert/util/vector.h>

#include <ert/enkf/local_ministep.hpp>
#include <ert/enkf/local_updatestep.hpp>
#include <ert/enkf/local_config.hpp>

#include "ert/python.hpp"
static auto logger = ert::get_logger("local_config");

/*

  +-------------------------- LocalUpdateStep ---------------------------------------+
  |                                                                                        |
  |                                                                                        |
  |    +----------------- local_ministep_type --------------------------------------+      |
  |    |                                                                            |      |
  |    |                                       /    +--- local_dataset_type ---+    |      |
  |    |                                       |    | PRESSURE                 |    |      |
  |    |                                       |    | SWAT                     |    |      |
  |    |                                       |    | SGAS                     |    |      |
  |    |                                       |    +--------------------------+    |      |
  |    |    +-- local_obsset_type ---+         |                                    |      |
  |    |    | WWCT:OP_2              |         |    +--- local_dataset_type ---+    |      |
  |    |    | WGOR:OP_1              |         |    | MULTFLT1                 |    |      |
  |    |    | RFT:WELL1              |  <------|    | MULTFLT2                 |    |      |
  |    |    | RFT:WELL3              |         |    | MULTFLT3                 |    |      |
  |    |    | WWCT:WELLX             |         |    +--------------------------+    |      |
  |    |    +------------------------+         |                                    |      |
  |    |                                       |    +--- local_dataset_type ---+    |      |
  |    |                                       |    | RELPERM1                 |    |      |
  |    |                                       |    | RELPERM2                 |    |      |
  |    |                                       |    | RELPERM3                 |    |      |
  |    |                                       \    +--------------------------+    |      |
  |    |                                                                            |      |
  |    +----------------------------------------------------------------------------+      |
  |                                                                                        |
  |                                                                                        |
  |    +----------------- local_ministep_type --------------------------------------+      |
  |    |                                                                            |      |
  |    |                                       /    +--- local_dataset_type ---+    |      |
  |    |    +-- local_obsset_type ---+         |    | PERMX PORO               |    |      |
  |    |    | 4D Seismic             |         |    | PRESSURE SWAT            |    |      |
  |    |    | Gravimetri             |         |    | SGAS                     |    |      |
  |    |    |                        |  <------|    +--------------------------+    |      |
  |    |    |                        |         |                                    |      |
  |    |    |                        |         |    +--- local_dataset_type ---+    |      |
  |    |    +------------------------+         |    | MULTFLT1                 |    |      |
  |    |                                       |    | MULTFLT2                 |    |      |
  |    |                                       |    | MULTFLT3                 |    |      |
  |    |                                       \    +--------------------------+    |      |
  |    |                                                                            |      |
  |    +----------------------------------------------------------------------------+      |
  |                                                                                        |
  +----------------------------------------------------------------------------------------+

This figure illustrates the different objects when configuring local
analysis:

LocalUpdateStep: This is is the top level configuration of the
   updating at one timestep. In principle you can have different
   updatestep configurations at the different timesteps, but it will
   typically be identical for all the time steps. Observe that the
   update at one time step can typically conist of several enkf
   updates, this is handled by using several local_ministep.

local_ministep_type: The ministep defines a collection of observations
   and state/parameter variables which are mutually dependant on
   eachother and should be updated together. The local_ministep will
   consist of *ONE* local_obsset of observations, and one or more
   local_dataset of data which should be updated.

local_obsset_type: This is a collection of observation data; there is
   exactly one local_obsset for each local_ministep.

local_dataset_type: This is a collection of data/parameters which
   should be updated together in the EnKF updating.


How the local_dataset_type is configured is quite important for the
core EnKF updating:

 1. All the members in one local_dataset instance are serialized and
    packed in the A-matrix together; i.e. in the example above the
    parameters RELPERM1,RELPERM2 and RELPERM3 are updated in one go.

 2. When using the standard EnKF the X matrix is calculated using
    the actual data vectors, and the results will be identical if we
    use one large local_dataset instance or several small. However
    when using more advanced techniques where the A matrix is used
    explicitly when calculating the update this will matter.

 3. If you have not entered a local configuration explicitly the
    default ALL_ACTIVE local configuration will be used.
*/

LocalConfig::LocalConfig(const std::vector<std::string> &parameter_keys,
                         const std::vector<std::string> &obs_keys)
    : m_global_updatestep("DEFAULT"),
      m_global_ministep(
          "ALL_ACTIVE") // The strings "ALL_ACTIVE" and "ALL_OBS" are part of the API.
      ,
      m_global_obsdata("ALL_OBS") {
    for (const auto &obs_key : obs_keys)
        this->m_global_obsdata.add_node(obs_key);

    for (const auto &param_key : parameter_keys)
        this->m_global_ministep.add_active_data(param_key.c_str());

    this->m_global_ministep.add_obsdata(this->m_global_obsdata);
    this->m_global_updatestep.add_ministep(this->m_global_ministep);
}

bool LocalConfig::has_obsdata(const std::string &obs_key) const {
    return this->m_obsdata.count(obs_key) == 1;
}

LocalMinistep &LocalConfig::make_ministep(const std::string &key) {
    if (this->m_ministep.count(key) == 1)
        return this->m_ministep.at(key);

    if (!this->m_updatestep.has_value())
        this->m_updatestep = LocalUpdateStep("DEFAULT");

    this->m_ministep.emplace(key, LocalMinistep(key.c_str()));
    auto &ministep = this->m_ministep.at(key);
    this->m_updatestep->add_ministep(ministep);
    printf("-------------------------------------------------------------------------------\n");
    printf("%s:  ministep: %p  \n", __func__, &ministep);
    return ministep;
}

LocalMinistep *LocalConfig::ministep(const std::string &key) {
    if (key == this->m_global_ministep.name())
        return this->global_ministep();
    else
        return &this->m_ministep.at(key);
}

LocalMinistep *LocalConfig::global_ministep() {
    return &this->m_global_ministep;
}

LocalObsData *LocalConfig::make_obsdata(const std::string &key) {
    if (this->m_obsdata.count(key) == 1)
        throw std::invalid_argument("Tried to add existing observation key");

    this->m_obsdata.emplace(key, LocalObsData(key));
    return &this->m_obsdata.at(key);
}

LocalObsData *LocalConfig::obsdata(const std::string &key) {
    if (key == this->m_global_obsdata.name())
        return this->global_obsdata();
    else
        return &this->m_obsdata.at(key);
}

LocalObsData *LocalConfig::global_obsdata() { return &this->m_global_obsdata; }

LocalUpdateStep &LocalConfig::updatestep() {
    if (this->m_updatestep.has_value())
        return this->m_updatestep.value();
    else
        return this->m_global_updatestep;
}

void LocalConfig::clear() const {
    logger->warning(
        "The LocalConfig::clear() function is deprecated and will be removed");
}

//# semeio functions:
//# Ert.getLocalConfig()
//# LocalConfig.createObsdata()
//# LocalConfig.createMinistep()
//# LocalConfig.getUpdatestep()
//# LocalMinistep.add
//# LocalMinistep.attachObsset()
//# LocalUpdateStep.attachMinistep()

RES_LIB_SUBMODULE("local.local_config", m) {
    py::class_<LocalConfig>(m, "LocalConfig")
        .def("createMinistep", &LocalConfig::make_ministep,
             py::return_value_policy::reference_internal)
        .def("createObsdata", &LocalConfig::make_obsdata,
             py::return_value_policy::reference_internal)
        .def("getUpdatestep", &LocalConfig::updatestep,
             py::return_value_policy::reference_internal)
        .def("global_ministep", &LocalConfig::global_ministep)
        .def("global_obsdata", &LocalConfig::global_obsdata)
        .def("getMinistep", &LocalConfig::ministep)
        .def("clear", &LocalConfig::clear)
        .def("getObsdata", &LocalConfig::obsdata);
}
