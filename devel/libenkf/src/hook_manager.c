/*
   Copyright (C) 2012  Statoil ASA, Norway.

   The file 'hook_manager.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <stdlib.h>
#include <string.h>

#include <ert/util/util.h>
#include <ert/util/subst_list.h>

#include <ert/config/config_parser.h>

#include <ert/job_queue/workflow.h>

#include <ert/enkf/config_keys.h>
#include <ert/enkf/hook_manager.h>
#include <ert/enkf/ert_workflow_list.h>
#include <ert/enkf/runpath_list.h>

#define HOOK_MANAGER_NAME             "HOOK MANAGER"
#define RUNPATH_LIST_FILE             ".ert_runpath_list"
#define QC_WORKFLOW_NAME              "QC WORKFLOW"
#define RUN_MODE_PRE_SIMULATION_NAME  "PRE_SIMULATION"
#define RUN_MODE_POST_SIMULATION_NAME "POST_SIMULATION"

struct hook_manager_struct {
  hook_workflow_type     * hook_workflow;      /* Will be generalized to a vector list */
  runpath_list_type      * runpath_list;
  ert_workflow_list_type * workflow_list;

  /* Deprecated stuff */
  hook_workflow_type     * post_hook_workflow; /* This is the good old QC workflow, kept for backward compatibility, obsolete */
  char                   * path;               /* The QC path, kept for backward compatibility */
};

hook_manager_type * hook_manager_alloc( ert_workflow_list_type * workflow_list , const char * path ) {
  hook_manager_type * hook_manager = util_malloc( sizeof * hook_manager );
  hook_manager->hook_workflow = hook_workflow_alloc(path);
  hook_manager->runpath_list = runpath_list_alloc( NULL );
  hook_manager->workflow_list = workflow_list;
  hook_manager_set_runpath_list_file( hook_manager, NULL, RUNPATH_LIST_FILE );

  /* Deprecated stuff */
  hook_manager->post_hook_workflow = hook_workflow_alloc(path);
  hook_manager->path = NULL;
  hook_manager_set_path( hook_manager , path );
  return hook_manager;
}

void hook_manager_init( hook_manager_type * hook_manager , const config_content_type * config) {
  if (config_content_has_item( config, RUNPATH_FILE_KEY))
      hook_manager_set_runpath_list_file(hook_manager, NULL, config_content_get_value(config, RUNPATH_FILE_KEY));
}

void hook_manager_free( hook_manager_type * hook_manager ) {
  runpath_list_free( hook_manager->runpath_list );
  free( hook_manager );
}

runpath_list_type * hook_manager_get_runpath_list( hook_manager_type * hook_manager ) {
  return hook_manager->runpath_list;
}

bool hook_manager_run_hook_workflow( const hook_manager_type * hook_manager , void * self) {
  const char * export_file = runpath_list_get_export_file( hook_manager->runpath_list );
  if (!util_file_exists( export_file ))
      fprintf(stderr,"** Warning: the file:%s with a list of runpath directories was not found - workflow will probably fail.\n" , export_file);

  return hook_workflow_run_workflow(hook_manager->hook_workflow, hook_manager->workflow_list, self);
}

bool hook_manager_has_hook_workflow( const hook_manager_type * hook_manager ) {
  return (hook_manager->hook_workflow != NULL);
}

const hook_workflow_type * hook_manager_get_hook_workflow( const hook_manager_type * hook_manager ) {
    return hook_manager->hook_workflow;
}

void hook_manager_init_hook( hook_manager_type * hook_manager , const config_content_type * config) {
  if (config_content_has_item( config , HOOK_WORKFLOW_KEY)) {
    const char * file_name = config_content_iget( config , HOOK_WORKFLOW_KEY, 0, 0 );
    char * workflow_name;
    util_alloc_file_components( file_name , NULL , &workflow_name , NULL );
    {
      workflow_type * workflow = ert_workflow_list_add_workflow( hook_manager->workflow_list , file_name , workflow_name);
      if (workflow != NULL) {
        ert_workflow_list_add_alias( hook_manager->workflow_list , workflow_name , HOOK_WORKFLOW_KEY );
        hook_workflow_set_workflow( hook_manager->hook_workflow , workflow);
      }
      hook_workflow_set_run_mode( hook_manager->hook_workflow , config_content_iget( config , HOOK_WORKFLOW_KEY, 0, 1 ));
    }
  }
}

void hook_manager_add_config_items( config_parser_type * config ) {
  config_schema_item_type * item;

  item = config_add_schema_item( config , QC_PATH_KEY , false  );
  config_schema_item_set_argc_minmax(item , 1 , 1 );

  item = config_add_schema_item( config , QC_WORKFLOW_KEY , false );
  config_schema_item_set_argc_minmax(item , 1 , 1 );
  config_schema_item_iset_type( item , 0 , CONFIG_EXISTING_PATH );

  item = config_add_schema_item( config , HOOK_WORKFLOW_KEY , false );
  config_schema_item_set_argc_minmax(item , 2 , 2 );
  config_schema_item_iset_type( item , 0 , CONFIG_EXISTING_PATH );
  config_schema_item_iset_type( item , 1 , CONFIG_STRING );

  item = config_add_schema_item( config , RUNPATH_FILE_KEY , false  );
  config_schema_item_set_argc_minmax(item , 1 , 1 );

}

void hook_manager_export_runpath_list( const hook_manager_type * hook_manager ) {
  runpath_list_fprintf( hook_manager->runpath_list );
}

const char * hook_manager_get_runpath_list_file( const hook_manager_type * hook_manager) {
  return runpath_list_get_export_file( hook_manager->runpath_list );
}

static void hook_manager_set_runpath_list_file__( hook_manager_type * hook_manager , const char * runpath_list_file) {
  runpath_list_set_export_file( hook_manager->runpath_list , runpath_list_file );
}

void hook_manager_set_runpath_list_file( hook_manager_type * hook_manager , const char * basepath, const char * filename) {

  if (filename && util_is_abs_path( filename ))
    hook_manager_set_runpath_list_file__( hook_manager , filename );
  else {
    const char * file = RUNPATH_LIST_FILE;

    if (filename != NULL)
      file = filename;

    char * file_with_path_prefix = NULL;
    if (basepath != NULL) {
      file_with_path_prefix = util_alloc_filename(basepath, file, NULL);
    }
    else
      file_with_path_prefix = util_alloc_string_copy(file);

    {
      char * absolute_path = util_alloc_abs_path(file_with_path_prefix);
      hook_manager_set_runpath_list_file__( hook_manager , absolute_path );
      free( absolute_path );
    }

    free(file_with_path_prefix);
  }
}

/*****************************************************************/
/* Deprecated stuff                                              */
/*****************************************************************/

void hook_manager_init_post_hook( hook_manager_type * hook_manager , const config_content_type * config) {
 if (config_content_has_item( config , QC_PATH_KEY ))
    hook_manager_set_path( hook_manager, config_content_get_value( config , QC_PATH_KEY ));

  if (config_content_has_item( config , QC_WORKFLOW_KEY)) {
    const char * file_name = config_content_get_value_as_path(config , QC_WORKFLOW_KEY);
    char * workflow_name;
    util_alloc_file_components( file_name , NULL , &workflow_name , NULL );
    {
      workflow_type * workflow = ert_workflow_list_add_workflow( hook_manager->workflow_list , file_name , workflow_name);
      if (workflow != NULL) {
        ert_workflow_list_add_alias( hook_manager->workflow_list , workflow_name , QC_WORKFLOW_NAME );
        hook_workflow_set_workflow( hook_manager->post_hook_workflow, workflow );
      }

      hook_workflow_set_run_mode( hook_manager->post_hook_workflow, RUN_MODE_POST_SIMULATION_NAME);
    }
  }
}

void hook_manager_set_path( hook_manager_type * hook_manager , const char * path) {
  hook_manager->path = util_realloc_string_copy( hook_manager->path , path );
}

const char * hook_manager_get_path( const hook_manager_type * hook_manager ) {
  return hook_manager->path;
}


const hook_workflow_type * hook_manager_get_post_hook_workflow( const hook_manager_type * hook_manager ) {
    return hook_manager->post_hook_workflow;
}

bool hook_manager_has_post_hook_workflow( const hook_manager_type * hook_manager ) {
  return (hook_manager->post_hook_workflow != NULL);
}

bool hook_manager_run_post_hook_workflow( const hook_manager_type * hook_manager , void * self) {
  const char * export_file = runpath_list_get_export_file( hook_manager->runpath_list );
  if (!util_file_exists( export_file ))
      fprintf(stderr,"** Warning: the file:%s with a list of runpath directories was not found - workflow will probably fail.\n" , export_file);

  return hook_workflow_run_workflow(hook_manager->post_hook_workflow, hook_manager->workflow_list, self);
}
