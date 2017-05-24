/*
   Copyright (C) 2017  Statoil ASA, Norway.

   The file 'enkf_config.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/enkf/enkf_config.h>
#include <ert/enkf/site_config.h>

struct enkf_config_struct {
  char * user_config_file;
  site_config_type * site_config;
};

static enkf_config_type * enkf_config_alloc_empty() {
  enkf_config_type * enkf_config = util_malloc(sizeof * enkf_config);
  enkf_config->user_config_file = NULL;

  enkf_config->site_config = NULL;

  return enkf_config;
}

enkf_config_type * enkf_config_alloc_load(const char * config_file) {
  enkf_config_type * enkf_config = enkf_config_alloc_empty(); 

  enkf_config->user_config_file = util_alloc_string_copy(config_file);
  enkf_config->site_config = site_config_alloc_load_user_config(config_file);

  return enkf_config;
}

void enkf_config_free(enkf_config_type * enkf_config) {
  site_config_free(enkf_config->site_config);

  free(enkf_config->user_config_file);
  free(enkf_config);
}

site_config_type * enkf_config_get_site_config(
                    const enkf_config_type * enkf_config
                    ) {
  return enkf_config->site_config;
}

const char * enkf_config_get_user_config_file(const enkf_config_type * enkf_config) {
  return enkf_config->user_config_file;
}
