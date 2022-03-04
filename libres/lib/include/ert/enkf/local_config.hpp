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

#include <ert/util/stringlist.h>

#include <ert/ecl/ecl_grid.h>

#include <ert/analysis/analysis_module.hpp>

#include <ert/enkf/local_updatestep.hpp>
#include <ert/enkf/local_ministep.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/enkf_obs.hpp>

typedef struct local_config_struct local_config_type;

LocalObsData *local_config_get_obsdata(const local_config_type *local_config,
                                       const char *key);
LocalObsData *local_config_alloc_obsdata_copy(local_config_type *local_config,
                                              const char *src_key,
                                              const char *target_key);

LocalObsData *local_config_get_obsdata(local_config_type *local_config,
                                       const char *key);

local_config_type *local_config_alloc();
extern "C" void local_config_clear(local_config_type *local_config);
extern "C" void local_config_clear_active(local_config_type *local_config);
extern "C" void local_config_free(local_config_type *local_config);
extern "C" local_ministep_type *
local_config_alloc_ministep(local_config_type *local_config, const char *key);
extern "C" local_updatestep_type *
local_config_get_updatestep(const local_config_type *local_config);
extern "C" local_ministep_type *
local_config_get_ministep(const local_config_type *local_config,
                          const char *key);
extern "C" bool local_config_has_obsdata(const local_config_type *local_config,
                                         const char *obsdata_name);
LocalObsData *local_config_alloc_obsdata(local_config_type *local_config,
                                         const char *obsdata_name);
bool local_config_has_obsdata(const local_config_type *local_config,
                              const char *obsdata_name);
#endif
