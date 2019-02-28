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

  queue_config_type *      site_config_get_queue_config(const site_config_type * site_config);
  const char *             site_config_get_location();
  bool                     site_config_has_queue_driver(const site_config_type * site_config, const char * driver_name);
  void                     site_config_update_pathvar(site_config_type * site_config, const char * pathvar, const char * value);

  const char *             site_config_get_config_file(const site_config_type*);

  const char *             site_config_get_manual_url( const site_config_type * site_config );
  void                     site_config_set_manual_url( site_config_type * site_config , const char * manual_url );
  const char *             site_config_get_default_browser( const site_config_type * site_config );
  void                     site_config_set_default_browser( site_config_type * site_config , const char * default_browser );

  const char            *  site_config_get_license_root_path( const site_config_type * site_config );
  void                     site_config_set_license_root_path( site_config_type * site_config , const char * license_root_path);

  void                     site_config_set_num_cpu( site_config_type * site_config , int num_cpu );
  void                     site_config_update_lsf_request(site_config_type *  , const forward_model_type *);
  void                     site_config_free(site_config_type *);
  ext_joblist_type       * site_config_get_installed_jobs( const site_config_type * );

  void                     site_config_set_max_running_lsf( site_config_type * site_config , int max_running_lsf);
  int                      site_config_get_max_running_lsf( const site_config_type * site_config );
  void                     site_config_set_max_running_rsh( site_config_type * site_config , int max_running_rsh);
  int                      site_config_get_max_running_rsh( const site_config_type * site_config);
  void                     site_config_set_max_running_local( site_config_type * site_config , int max_running_local);
  int                      site_config_get_max_running_local( const site_config_type * site_config );
  const env_varlist_type * site_config_get_env_varlist(const site_config_type * site_config);
  void                     site_config_clear_pathvar( site_config_type * site_config );
  stringlist_type        * site_config_get_path_variables( const site_config_type * site_config );
  stringlist_type        * site_config_get_path_values( const site_config_type * site_config );

  void                     site_config_clear_rsh_host_list( site_config_type * site_config );
  hash_type              * site_config_get_rsh_host_list( const site_config_type * site_config );
  void                     site_config_add_rsh_host( site_config_type * site_config , const char * rsh_host , int max_running);

  void                     site_config_set_lsf_queue( site_config_type * site_config , const char * lsf_queue);
  const char             * site_config_get_lsf_queue( const site_config_type * site_config );
  void                     site_config_set_lsf_request( site_config_type * site_config , const char * lsf_request);
  const char             * site_config_get_lsf_request( const site_config_type * site_config );

  const char             * site_config_get_job_queue_name( const site_config_type * site_config );
  void                     site_config_set_job_queue( site_config_type * site_config , const char * queue_name );

  void                     site_config_set_rsh_command( site_config_type * site_config , const char * rsh_command);
  const char             * site_config_get_rsh_command( const site_config_type * site_config );
  void                     site_config_set_ext_job_search_path(site_config_type * site_config, bool search_path);


  bool                     site_config_set_job_script( site_config_type * site_config , const char * job_script );

  void                     site_config_set_max_submit( site_config_type * site_config , int max_submit );
  int                      site_config_get_max_submit(const site_config_type * site_config );

  bool                     site_config_queue_is_running( const site_config_type * site_config );
  int                      site_config_install_job(site_config_type * site_config , const char * job_name , const char * install_file);
  bool                     site_config_del_job( site_config_type * site_config , const char * job_name );

  void                     site_config_set_umask( site_config_type * site_config , mode_t umask);
  mode_t                   site_config_get_umask( const site_config_type * site_config );

  site_config_type       * site_config_alloc_load_user_config(const char *);
  site_config_type       * site_config_alloc(const config_content_type * config_content);

  config_content_type    * site_config_alloc_content(config_parser_type*);

  void                     site_config_add_config_items( config_parser_type * config , bool site_mode);
#ifdef __cplusplus
}
#endif
#endif
