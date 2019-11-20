/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'site_config.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_SITE_CONFIG_H
#define ERT_SITE_CONFIG_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include <ert/util/stringlist.h>

#include <ert/config/config_parser.hpp>
#include <ert/config/config_content.hpp>

#include <ert/job_queue/job_queue.hpp>
#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/forward_model.hpp>
#include <ert/job_queue/environment_varlist.hpp>

#include <ert/enkf/queue_config.hpp>


typedef struct site_config_struct site_config_type;

  const char *             site_config_get_location();
  const char *             site_config_get_config_file(const site_config_type*);
  const char            *  site_config_get_license_root_path( const site_config_type * site_config );
  void                     site_config_set_license_root_path( site_config_type * site_config , const char * license_root_path);
  void                     site_config_free(site_config_type *);
  ext_joblist_type       * site_config_get_installed_jobs( const site_config_type * );
  const env_varlist_type * site_config_get_env_varlist(const site_config_type * site_config);
  int                      site_config_install_job(site_config_type * site_config , const char * job_name , const char * install_file);
  void                     site_config_set_umask( site_config_type * site_config , mode_t umask);
  mode_t                   site_config_get_umask( const site_config_type * site_config );
  site_config_type       * site_config_alloc_load_user_config(const char *);
  site_config_type       * site_config_alloc(const config_content_type * config_content);
  site_config_type       * site_config_alloc_full(ext_joblist_type * ext_joblist, env_varlist_type * env_varlist, int umask);
  config_content_type    * site_config_alloc_content(config_parser_type*);
  void                     site_config_add_config_items( config_parser_type * config , bool site_mode);
#ifdef __cplusplus
}
#endif
#endif
