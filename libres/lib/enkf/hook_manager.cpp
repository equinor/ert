/*
   Copyright (C) 2012  Equinor ASA, Norway.

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

#include <ert/util/util.h>
#include <ert/util/vector.h>
#include <ert/res_util/subst_list.hpp>

#include <ert/config/config_parser.hpp>

#include <ert/job_queue/workflow.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/site_config.hpp>
#include <ert/enkf/hook_manager.hpp>

struct hook_manager_struct {
  vector_type            * hook_workflow_list;  /* vector of hook_workflow_type instances */
  runpath_list_type      * runpath_list;
  ert_workflow_list_type * workflow_list;
  hash_type              * input_context;
};

hook_manager_type * hook_manager_alloc(
                    ert_workflow_list_type * workflow_list,
                    const config_content_type * config_content) {

  hook_manager_type * hook_manager = hook_manager_alloc_default(workflow_list);

  if(config_content)
    hook_manager_init(hook_manager, config_content);

  return hook_manager;
}

void hook_manager_free( hook_manager_type * hook_manager ) {
  if (hook_manager->runpath_list)
    runpath_list_free( hook_manager->runpath_list );

  vector_free( hook_manager->hook_workflow_list );
  hash_free( hook_manager->input_context );
  free( hook_manager );
}


runpath_list_type * hook_manager_get_runpath_list(const hook_manager_type * hook_manager) {
  return hook_manager->runpath_list;
}


static void hook_manager_add_workflow( hook_manager_type * hook_manager , const char * workflow_name , hook_run_mode_enum run_mode) {
  if (ert_workflow_list_has_workflow( hook_manager->workflow_list , workflow_name) ){
    workflow_type * workflow = ert_workflow_list_get_workflow( hook_manager->workflow_list , workflow_name);
    hook_workflow_type * hook = hook_workflow_alloc( workflow , run_mode );
    vector_append_owned_ref(hook_manager->hook_workflow_list, hook , hook_workflow_free__);
  }
  else {
    fprintf(stderr, "** Warning: While hooking workflow: %s not recognized among the list of loaded workflows.", workflow_name);
  }
}

hook_manager_type * hook_manager_alloc_default(ert_workflow_list_type * workflow_list) {
  hook_manager_type * hook_manager = (hook_manager_type *)util_malloc( sizeof * hook_manager );
  hook_manager->workflow_list = workflow_list;
  
  hook_manager->hook_workflow_list = vector_alloc_new();

  config_parser_type * config = config_alloc();
  config_content_type * site_config_content = site_config_alloc_content(config);

  if (config_content_has_item( site_config_content , HOOK_WORKFLOW_KEY)) {
    for (int ihook = 0; ihook < config_content_get_occurences(site_config_content , HOOK_WORKFLOW_KEY); ihook++) {
      const char * workflow_name = config_content_iget( site_config_content , HOOK_WORKFLOW_KEY, ihook , 0 );
      hook_run_mode_enum run_mode = hook_workflow_run_mode_from_name(config_content_iget(site_config_content , HOOK_WORKFLOW_KEY , ihook , 1));
      hook_manager_add_workflow( hook_manager , workflow_name , run_mode );
    }
  }
  config_free(config);
  config_content_free(site_config_content);

  hook_manager->runpath_list = NULL;

  hook_manager->input_context = hash_alloc();

  return hook_manager;
}

hook_manager_type * hook_manager_alloc_full(
        ert_workflow_list_type * workflow_list,
        const char * runpath_list_file,
        const char ** hook_workflow_names,
        const char ** hook_workflow_run_modes,
        int hook_workflow_count) {

    hook_manager_type * hook_manager = hook_manager_alloc_default(workflow_list);

    for (int i = 0; i < hook_workflow_count; ++i) {
      const char * workflow_name = hook_workflow_names[i];
      hook_run_mode_enum run_mode = hook_workflow_run_mode_from_name(hook_workflow_run_modes[i]);
      hook_manager_add_workflow( hook_manager , workflow_name , run_mode );
    }

    hook_manager->runpath_list = runpath_list_alloc(runpath_list_file);

    return hook_manager;
}

void hook_manager_init( hook_manager_type * hook_manager , const config_content_type * config_content) {
  if (config_content_has_item( config_content , HOOK_WORKFLOW_KEY)) {
    for (int ihook = 0; ihook < config_content_get_occurences(config_content , HOOK_WORKFLOW_KEY); ihook++) {
      const char * workflow_name = config_content_iget( config_content , HOOK_WORKFLOW_KEY, ihook , 0 );
      hook_run_mode_enum run_mode = hook_workflow_run_mode_from_name(config_content_iget(config_content , HOOK_WORKFLOW_KEY , ihook , 1));
      hook_manager_add_workflow( hook_manager , workflow_name , run_mode );
    }
  }

  {
    char * runpath_list_file;

    if (config_content_has_item(config_content, RUNPATH_FILE_KEY))
      runpath_list_file = util_alloc_string_copy( config_content_get_value_as_abspath(config_content, RUNPATH_FILE_KEY));
    else
      runpath_list_file = util_alloc_filename( config_content_get_config_path(config_content), RUNPATH_LIST_FILE, NULL);

    hook_manager->runpath_list = runpath_list_alloc(runpath_list_file);
    free( runpath_list_file);
  }
}



void hook_manager_add_config_items( config_parser_type * config ) {
  config_schema_item_type * item;

  item = config_add_schema_item( config , HOOK_WORKFLOW_KEY , false );
  config_schema_item_set_argc_minmax(item , 2 , 2 );
  config_schema_item_iset_type( item , 0 , CONFIG_STRING );
  config_schema_item_iset_type( item , 1 , CONFIG_STRING );
  {
    stringlist_type * argv = stringlist_alloc_new();

    stringlist_append_copy(argv, RUN_MODE_PRE_SIMULATION_NAME);
    stringlist_append_copy(argv, RUN_MODE_POST_SIMULATION_NAME);
    stringlist_append_copy(argv, RUN_MODE_PRE_UPDATE_NAME);
    stringlist_append_copy(argv, RUN_MODE_POST_UPDATE_NAME);
    config_schema_item_set_indexed_selection_set(item, 1, argv);

    stringlist_free( argv );
  }

  item = config_add_schema_item( config , RUNPATH_FILE_KEY , false  );
  config_schema_item_set_argc_minmax(item , 1 , 1 );
  config_schema_item_iset_type(item, 0, CONFIG_PATH);
}


const char * hook_manager_get_runpath_list_file( const hook_manager_type * hook_manager) {
  return runpath_list_get_export_file( hook_manager->runpath_list );
}



void hook_manager_run_workflows( const hook_manager_type * hook_manager , hook_run_mode_enum run_mode , void * self )
{
  bool verbose = false;
  for (int i=0; i < vector_get_size( hook_manager->hook_workflow_list ); i++) {
    hook_workflow_type * hook_workflow = (hook_workflow_type *)vector_iget( hook_manager->hook_workflow_list , i );
    if (hook_workflow_get_run_mode(hook_workflow) == run_mode) {
      workflow_type * workflow = hook_workflow_get_workflow( hook_workflow );
      workflow_run( workflow, self , verbose , ert_workflow_list_get_context( hook_manager->workflow_list ));
      /*
        The workflow_run function will return a bool to indicate
        success/failure, and in the case of error the function
        workflow_get_last_error() can be used to get a config_error
        object.
      */
    }
  }
}

const hook_workflow_type * hook_manager_iget_hook_workflow(const hook_manager_type * hook_manager, int index){
 return (hook_workflow_type *) vector_iget(hook_manager->hook_workflow_list, index);
}

int hook_manager_get_size(const hook_manager_type * hook_manager){
 return vector_get_size(hook_manager->hook_workflow_list);
}
