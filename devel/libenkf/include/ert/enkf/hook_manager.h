/*
   Copyright (C) 2012  Statoil ASA, Norway.

   The file 'hook_manager.h' is part of ERT - Ensemble based Reservoir Tool.

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
#ifndef __HOOK_MANAGER_H__
#define __HOOK_MANAGER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <ert/config/config_parser.h>

#include <ert/enkf/ert_workflow_list.h>
#include <ert/enkf/runpath_list.h>

  typedef struct hook_manager_struct hook_manager_type;

  bool                  hook_manager_has_workflow( const hook_manager_type * hook_manager );
  const workflow_type * hook_manager_get_workflow( const hook_manager_type * hook_manager );
  hook_manager_type      * hook_manager_alloc(ert_workflow_list_type * workflow_list ,  const char * path);
  void                  hook_manager_free();
  bool                  hook_manager_run_workflow( const hook_manager_type * hook_manager , void * self);
  runpath_list_type   * hook_manager_get_runpath_list( hook_manager_type * hook_manager );
  void                  hook_manager_set_path( hook_manager_type * hook_manager , const char * path);
  const char          * hook_manager_get_path( const hook_manager_type * hook_manager );
  void                  hook_manager_init( hook_manager_type * hook_manager , const config_content_type * config);
  void                  hook_manager_export_runpath_list( const hook_manager_type * hook_manager );
  void                  hook_manager_add_config_items( config_parser_type * config );
  void                  hook_manager_set_runpath_list_file( hook_manager_type * hook_manager , const char * path, const char * filename);
  const char          * hook_manager_get_runpath_list_file(const hook_manager_type * hook_manager);



#ifdef __cplusplus
}
#endif
#endif
