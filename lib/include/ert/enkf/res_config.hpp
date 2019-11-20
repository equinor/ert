/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'res_config.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_RES_CONFIG_H
#define ERT_RES_CONFIG_H

#include <ert/config/config_settings.hpp>

#include <ert/enkf/site_config.hpp>
#include <ert/enkf/rng_config.hpp>
#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/ert_workflow_list.hpp>
#include <ert/enkf/subst_config.hpp>
#include <ert/enkf/hook_manager.hpp>
#include <ert/enkf/ert_template.hpp>
#include <ert/enkf/ecl_config.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/enkf/log_config.hpp>
#include <ert/enkf/log_config.hpp>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct res_config_struct res_config_type;

  void res_config_init_config_parser(config_parser_type * config_parser);
  res_config_type * res_config_alloc_load(const char *);
  res_config_type * res_config_alloc(const config_content_type *);

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
                                        queue_config_type * queue_config);
  void              res_config_free(res_config_type *);
  void              res_config_add_config_items(config_parser_type * config_parser);

config_content_type * res_config_alloc_user_content(const char * user_config_file,
                                                    config_parser_type * config_parser);
const site_config_type       * res_config_get_site_config(const res_config_type *);
rng_config_type              * res_config_get_rng_config(const res_config_type *);
const analysis_config_type   * res_config_get_analysis_config(const res_config_type *);
ert_workflow_list_type       * res_config_get_workflow_list(const res_config_type *);
subst_config_type            * res_config_get_subst_config(const res_config_type * res_config);
const hook_manager_type      * res_config_get_hook_manager(const res_config_type * res_config);
ert_templates_type           * res_config_get_templates(const res_config_type * res_config);
const ecl_config_type        * res_config_get_ecl_config(const res_config_type * res_config);
ensemble_config_type         * res_config_get_ensemble_config(const res_config_type * res_config);
model_config_type            * res_config_get_model_config(const res_config_type * res_config);
const log_config_type        * res_config_get_log_config(const res_config_type * res_config);
queue_config_type            * res_config_get_queue_config(const res_config_type * res_config);

const char * res_config_get_config_directory(const res_config_type *);
const char * res_config_get_user_config_file(const res_config_type *);

#ifdef __cplusplus
}
#endif
#endif
