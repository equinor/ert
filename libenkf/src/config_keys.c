/*
   Copyright (C) 2017  Statoil ASA, Norway.

   The file 'config_keys.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/enkf/config_keys.h>

const char * config_keys_get_config_directory_key() {
  return CONFIG_DIRECTORY_KEY;
}

const char * config_keys_get_queue_system_key() {
  return QUEUE_SYSTEM_KEY;
}

const char * config_keys_get_run_template_key() {
  return RUN_TEMPLATE_KEY;
}

const char * config_keys_get_gen_kw_key() {
  return GEN_KW_KEY;
}

const char * config_keys_get_queue_option_key() {
  return QUEUE_OPTION_KEY;
}

const char * config_keys_get_install_job_key() {
  return INSTALL_JOB_KEY;
}

const char * config_keys_get_plot_setting_key() {
  return PLOT_SETTING_KEY;
}
