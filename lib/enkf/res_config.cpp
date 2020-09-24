/*
   Copyright (C) 2017  Equinor ASA, Norway.

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

#include <ert/res_util/subst_func.hpp>
#include <ert/res_util/subst_list.hpp>

#include <ert/config/config_settings.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/res_config.hpp>

struct res_config_struct {

  char * user_config_file;
  char * config_dir;

  site_config_type       * site_config;
  rng_config_type        * rng_config;
  analysis_config_type   * analysis_config;
  ert_workflow_list_type * workflow_list;
  subst_config_type      * subst_config;
  hook_manager_type      * hook_manager;
  ert_templates_type     * templates;
  ecl_config_type        * ecl_config;
  ensemble_config_type   * ensemble_config;
  model_config_type      * model_config;
  log_config_type        * log_config;
  queue_config_type      * queue_config;

};


static char * res_config_alloc_config_directory(const char * user_config_file);


static res_config_type * res_config_alloc_empty() {
  res_config_type * res_config = (res_config_type *)util_malloc(sizeof * res_config);
  res_config->user_config_file  = NULL;
  res_config->config_dir        = NULL;

  res_config->site_config       = NULL;
  res_config->rng_config        = NULL;
  res_config->analysis_config   = NULL;
  res_config->workflow_list     = NULL;
  res_config->subst_config      = NULL;
  res_config->hook_manager      = NULL;
  res_config->templates         = NULL;
  res_config->ecl_config        = NULL;
  res_config->ensemble_config   = NULL;
  res_config->model_config      = NULL;
  res_config->log_config        = NULL;
  res_config->queue_config      = NULL;

  return res_config;
}


void res_config_add_config_items(config_parser_type * config_parser) {
  config_add_key_value(config_parser, RES_CONFIG_FILE_KEY,   false, CONFIG_EXISTING_PATH);
  config_add_key_value(config_parser, CONFIG_DIRECTORY_KEY, false, CONFIG_EXISTING_PATH);
}


void res_config_init_config_parser(config_parser_type * config_parser) {
  model_config_init_config_parser(config_parser);
  res_config_add_config_items(config_parser);
}


static void res_config_install_config_key(
                     config_parser_type * config_parser,
                     config_content_type * config_content,
                     const char * key, const char * value
                     ) {

  config_schema_item_type * schema_item = config_get_schema_item(config_parser, key);

  if(!config_content_has_item(config_content, key))
    config_content_add_item(config_content, schema_item, NULL);

  config_content_item_type * content_item = config_content_get_item(config_content, key);

  config_content_node_type * new_node = config_content_item_alloc_node(content_item, NULL);
  config_content_node_add_value(new_node, value);

  if(!new_node)
    util_abort(
            "%s: Failed to internally install %s: %s\n",
            __func__, key, value
            );

  config_content_add_node(config_content, new_node);
}


config_content_type * res_config_alloc_user_content(
                                const char * user_config_file,
                                config_parser_type * config_parser) {

  if(!user_config_file)
    return NULL;

  // Read config file
  config_content_type * config_content = model_config_alloc_content(
                                                     user_config_file,
                                                     config_parser
                                                     );

  res_config_add_config_items(config_parser);

  // Install config file name
  char * res_config_file = (user_config_file ?
                                    util_alloc_realpath(user_config_file) :
                                    NULL
                                    );

  res_config_install_config_key(config_parser,
                                config_content,
                                RES_CONFIG_FILE_KEY,
                                res_config_file
                                );

  // Install working directory
  char * res_config_dir = res_config_alloc_config_directory(res_config_file);

  res_config_install_config_key(config_parser,
                                config_content,
                                CONFIG_DIRECTORY_KEY,
                                res_config_dir
                                );

  free(res_config_file);
  free(res_config_dir);

  return config_content;
}


res_config_type * res_config_alloc_load(const char * config_file) {
  config_parser_type * config_parser   = config_alloc();
  config_content_type * config_content = res_config_alloc_user_content(config_file, config_parser);

  res_config_type * res_config = res_config_alloc(config_content);

  config_content_free(config_content);
  config_free(config_parser);

  return res_config;
}


static void res_config_init(
        res_config_type * res_config,
        const config_content_type * config_content)
{
  if(config_content_has_item(config_content, RES_CONFIG_FILE_KEY)) {
    const char * res_config_file = config_content_get_value_as_abspath(config_content, RES_CONFIG_FILE_KEY);
    res_config->user_config_file = util_alloc_string_copy(res_config_file);
  }

  if(config_content_has_item(config_content, CONFIG_DIRECTORY_KEY)) {
    const char * config_dir = config_content_get_value_as_abspath(config_content, CONFIG_DIRECTORY_KEY);
    res_config->config_dir = util_alloc_string_copy(config_dir);
  }
}


res_config_type * res_config_alloc(const config_content_type * config_content) {
  res_config_type * res_config = res_config_alloc_empty();

  if(config_content)
    res_config_init(res_config, config_content);

  res_config->subst_config    = subst_config_alloc(config_content);
  res_config->site_config     = site_config_alloc(config_content);
  res_config->rng_config      = rng_config_alloc(config_content);
  res_config->analysis_config = analysis_config_alloc(config_content);

  res_config->workflow_list   = ert_workflow_list_alloc(
                                    subst_config_get_subst_list(res_config->subst_config),
                                    config_content
                                    );

  res_config->hook_manager    = hook_manager_alloc(
                                    res_config->workflow_list,
                                    config_content
                                    );

  res_config->templates       = ert_templates_alloc(
                                    subst_config_get_subst_list(res_config->subst_config),
                                    config_content
                                    );

  res_config->ecl_config      = ecl_config_alloc(config_content);

  res_config->ensemble_config = ensemble_config_alloc(config_content,
                                            ecl_config_get_grid(res_config->ecl_config),
                                            ecl_config_get_refcase(res_config->ecl_config)
                                    );

  res_config->model_config    = model_config_alloc(config_content,
                                                   res_config->config_dir,
                                                   site_config_get_installed_jobs(res_config->site_config),
                                                   ecl_config_get_last_history_restart(res_config->ecl_config),
                                                   ecl_config_get_refcase(res_config->ecl_config)
                                   );

  res_config->log_config      = log_config_alloc(config_content);

  res_config->queue_config = queue_config_alloc(config_content);

  return res_config;
}

res_config_type * res_config_alloc_full(char * config_dir,
                                        char * user_config_file,
                                        subst_config_type * subst_config,
                                        site_config_type * site_config,
                                        rng_config_type * rng_config,
                                        analysis_config_type * analysis_config,
                                        ert_workflow_list_type * workflow_list,
                                        hook_manager_type * hook_manager,
                                        ert_templates_type * templates,
                                        ecl_config_type * ecl_config,
                                        ensemble_config_type * ensemble_config,
                                        model_config_type * model_config,
                                        log_config_type * log_config,
                                        queue_config_type * queue_config){
  res_config_type  * res_config = res_config_alloc_empty();

  res_config->user_config_file = util_alloc_string_copy(user_config_file);
  res_config->config_dir = util_alloc_string_copy(config_dir);
  res_config->subst_config = subst_config;
  res_config->site_config = site_config;
  res_config->rng_config = rng_config;
  res_config->analysis_config = analysis_config;
  res_config->workflow_list = workflow_list;
  res_config->hook_manager = hook_manager;
  res_config->templates = templates;
  res_config->ecl_config = ecl_config;
  res_config->ensemble_config = ensemble_config;
  res_config->model_config = model_config;
  res_config->log_config = log_config;
  res_config->queue_config = queue_config;
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
  hook_manager_free(res_config->hook_manager);
  ert_templates_free(res_config->templates);
  ecl_config_free(res_config->ecl_config);
  ensemble_config_free(res_config->ensemble_config);
  model_config_free(res_config->model_config);
  log_config_free(res_config->log_config);

  free(res_config->user_config_file);
  free(res_config->config_dir);
  queue_config_free(res_config->queue_config);
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

const hook_manager_type * res_config_get_hook_manager(
                    const res_config_type * res_config
                   ) {
  return res_config->hook_manager;
}

ert_templates_type * res_config_get_templates(
                    const res_config_type * res_config
                  ) {
  return res_config->templates;
}


const ecl_config_type * res_config_get_ecl_config(
                    const res_config_type * res_config
                  ) {
  return res_config->ecl_config;
}

ensemble_config_type * res_config_get_ensemble_config(
                    const res_config_type * res_config
                  ) {
  return res_config->ensemble_config;
}

model_config_type * res_config_get_model_config(
                    const res_config_type * res_config
                  ) {
  return res_config->model_config;
}

const log_config_type * res_config_get_log_config(
                    const res_config_type * res_config
                  ) {
  return res_config->log_config;
}

queue_config_type * res_config_get_queue_config(
                    const res_config_type * res_config
                    ){
  return res_config->queue_config;
}

static char * res_config_alloc_config_directory(const char * user_config_file) {
  if(user_config_file == NULL)
    return NULL;

  char * path = NULL;
  char * realpath = util_alloc_link_target(user_config_file);
  char * abspath  = util_alloc_realpath(realpath);
  util_alloc_file_components(abspath, &path, NULL, NULL);
  free(realpath);
  free(abspath);

  return path;
}

const char * res_config_get_config_directory(const res_config_type * res_config) {
  return res_config->config_dir;
}

const char * res_config_get_user_config_file(const res_config_type * res_config) {
  return res_config->user_config_file;
}
