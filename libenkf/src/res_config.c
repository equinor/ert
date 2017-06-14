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

#include <ert/util/subst_list.h>
#include <ert/util/subst_func.h>

#include <ert/enkf/res_config.h>
#include <ert/enkf/site_config.h>
#include <ert/enkf/rng_config.h>
#include <ert/enkf/analysis_config.h>
#include <ert/enkf/ert_workflow_list.h>
#include <ert/enkf/subst_config.h>

struct res_config_struct {

  char * user_config_file;
  char * working_dir;

  site_config_type       * site_config;
  rng_config_type        * rng_config;
  analysis_config_type   * analysis_config;
  ert_workflow_list_type * workflow_list;
  subst_config_type      * subst_config;

};

static char * res_config_alloc_working_directory(const char * user_config_file);

static res_config_type * res_config_alloc_empty() {
  res_config_type * res_config = util_malloc(sizeof * res_config);
  res_config->user_config_file = NULL;
  res_config->working_dir      = NULL;

  res_config->site_config       = NULL;
  res_config->rng_config        = NULL;
  res_config->analysis_config   = NULL;
  res_config->workflow_list     = NULL;
  res_config->subst_config      = NULL;

  return res_config;
}

res_config_type * res_config_alloc_load(const char * config_file) {
  res_config_type * res_config = res_config_alloc_empty(); 
  res_config->user_config_file = util_alloc_string_copy(config_file);
  res_config->working_dir      = res_config_alloc_working_directory(config_file);

  res_config->subst_config    = subst_config_alloc_load(config_file, res_config->working_dir);

  res_config->site_config     = site_config_alloc_load_user_config(config_file);
  res_config->rng_config      = rng_config_alloc_load_user_config(config_file);
  res_config->analysis_config = analysis_config_alloc_load(config_file);
  res_config->workflow_list   = ert_workflow_list_alloc_load(
                                    subst_config_get_subst_list(res_config->subst_config),
                                    config_file
                                    );

  return res_config;
}

void res_config_free(res_config_type * res_config) {
  if(!res_config)
    return;

  site_config_free(res_config->site_config);
  rng_config_free(res_config->rng_config);
  analysis_config_free(res_config->analysis_config);
  ert_workflow_list_free(res_config->workflow_list);
  subst_config_free(res_config->subst_config);

  free(res_config->user_config_file);
  free(res_config->working_dir);
  free(res_config);
}

const site_config_type * res_config_get_site_config(
                    const res_config_type * res_config
                    ) {
  return res_config->site_config;
}

rng_config_type * res_config_get_rng_config(
                    const res_config_type * res_config
                    ) {
  return res_config->rng_config;
}

const analysis_config_type * res_config_get_analysis_config(
                    const res_config_type * res_config
                    ) {
  return res_config->analysis_config;
}

ert_workflow_list_type * res_config_get_workflow_list(
                    const res_config_type * res_config
        ) {
  return res_config->workflow_list;
}

subst_config_type * res_config_get_subst_config(
                    const res_config_type * res_config
                   ) {
  return res_config->subst_config;
}

static char * res_config_alloc_working_directory(const char * user_config_file) {
  if(user_config_file == NULL)
    return NULL;

  char * path = NULL;
  char * realpath = util_alloc_link_target(user_config_file);
  char * abspath  = util_alloc_abs_path(realpath);
  util_alloc_file_components(abspath, &path, NULL, NULL);
  free(realpath);
  free(abspath);

  return path;
}

const char * res_config_get_working_directory(const res_config_type * res_config) {
  return res_config->working_dir;
}

const char * res_config_get_user_config_file(const res_config_type * res_config) {
  return res_config->user_config_file;
}

const char * res_config_get_site_config_file(const res_config_type * res_config) {
  return site_config_get_config_file(res_config->site_config);
}
