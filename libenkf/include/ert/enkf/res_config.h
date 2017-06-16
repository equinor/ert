/*
   Copyright (C) 2017  Statoil ASA, Norway.

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

#include <ert/util/subst_list.h>
#include <ert/util/subst_func.h>

#include <ert/config/config_settings.h>

#include <ert/enkf/site_config.h>
#include <ert/enkf/rng_config.h>
#include <ert/enkf/analysis_config.h>
#include <ert/enkf/ert_workflow_list.h>
#include <ert/enkf/subst_config.h>
#include <ert/enkf/hook_manager.h>
#include <ert/enkf/ert_template.h>
#include <ert/enkf/ecl_config.h>
#include <ert/enkf/ensemble_config.h>

typedef struct res_config_struct res_config_type;

res_config_type * res_config_alloc_load(const char *);
void              res_config_free(res_config_type *);

const site_config_type       * res_config_get_site_config(const res_config_type *);
rng_config_type              * res_config_get_rng_config(const res_config_type *);
const analysis_config_type   * res_config_get_analysis_config(const res_config_type *);
ert_workflow_list_type       * res_config_get_workflow_list(const res_config_type *);
subst_config_type            * res_config_get_subst_config(const res_config_type * res_config);
const hook_manager_type      * res_config_get_hook_manager(const res_config_type * res_config);
ert_templates_type           * res_config_get_templates(const res_config_type * res_config);
const config_settings_type   * res_config_get_plot_config(const res_config_type * res_config);
const ecl_config_type        * res_config_get_ecl_config(const res_config_type * res_config);
ensemble_config_type         * res_config_get_ensemble_config(const res_config_type * res_config);

const char * res_config_get_working_directory(const res_config_type *);
const char * res_config_get_user_config_file(const res_config_type *);
const char * res_config_get_site_config_file(const res_config_type *);

#endif
