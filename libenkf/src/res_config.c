/*
   Copyright (C) 2017  Statoil ASA, Norway.

   The file 'res_config.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/enkf/res_config.h>
#include <ert/enkf/site_config.h>
#include <ert/enkf/rng_config.h>

struct res_config_struct {
  char * user_config_file;
  site_config_type * site_config;
  rng_config_type  * rng_config;
};

static res_config_type * res_config_alloc_empty() {
  res_config_type * res_config = util_malloc(sizeof * res_config);
  res_config->user_config_file = NULL;

  res_config->site_config = NULL;
  res_config->rng_config  = NULL;

  return res_config;
}

res_config_type * res_config_alloc_load(const char * config_file) {
  res_config_type * res_config = res_config_alloc_empty(); 

  res_config->user_config_file = util_alloc_string_copy(config_file);

  res_config->site_config = site_config_alloc_load_user_config(config_file);
  res_config->rng_config  = rng_config_alloc_load_user_config(config_file);

  return res_config;
}

void res_config_free(res_config_type * res_config) {
  site_config_free(res_config->site_config);
  rng_config_free(res_config->rng_config);

  free(res_config->user_config_file);
  free(res_config);
}

site_config_type * res_config_get_site_config(
                    const res_config_type * res_config
                    ) {
  return res_config->site_config;
}

rng_config_type * res_config_get_rng_config(
                    const res_config_type * res_config
                    ) {
  return res_config->rng_config;
}

const char * res_config_get_user_config_file(const res_config_type * res_config) {
  return res_config->user_config_file;
}
