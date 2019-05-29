/*
   Copyright (C) 2012  Equinor ASA, Norway.

   The file 'ert_workflow_list.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

#include <set>
#include <string>

#include <ert/util/hash.h>
#include <ert/util/ecl_version.h>
#include <ert/util/stringlist.h>
#include <ert/util/util.h>
#include <ert/util/type_macros.h>
#include <ert/res_util/subst_list.hpp>

#include <ert/config/config_parser.hpp>
#include <ert/config/config_error.hpp>
#include <ert/config/config_schema_item.hpp>

#include <ert/job_queue/workflow.hpp>
#include <ert/job_queue/workflow_job.hpp>
#include <ert/job_queue/workflow_joblist.hpp>

#include <ert/res_util/res_log.hpp>

#include <ert/enkf/ert_workflow_list.hpp>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/site_config.hpp>
#include <ert/enkf/model_config.hpp>

#define ERT_WORKFLOW_LIST_TYPE_ID 8856275

struct ert_workflow_list_struct {
  UTIL_TYPE_ID_DECLARATION;
  hash_type               * workflows;
  hash_type               * alias_map;
  workflow_joblist_type   * joblist;
  const subst_list_type   * context;
  const config_error_type * last_error;
  bool                      verbose;
};

static void ert_workflow_list_init(ert_workflow_list_type * workflow_list, const config_content_type * config);

ert_workflow_list_type * ert_workflow_list_alloc_empty(const subst_list_type * context) {
  ert_workflow_list_type * workflow_list = (ert_workflow_list_type *)util_malloc( sizeof * workflow_list );
  UTIL_TYPE_ID_INIT( workflow_list , ERT_WORKFLOW_LIST_TYPE_ID );
  workflow_list->workflows  = hash_alloc();
  workflow_list->alias_map  = hash_alloc();
  workflow_list->joblist    = workflow_joblist_alloc();
  workflow_list->context    = context;
  workflow_list->last_error = NULL;
  ert_workflow_list_set_verbose( workflow_list , DEFAULT_WORKFLOW_VERBOSE );
  return workflow_list;
}

ert_workflow_list_type * ert_workflow_list_alloc_load_site_config(const subst_list_type * context) {
  ert_workflow_list_type * workflow_list = ert_workflow_list_alloc_empty(context);

  config_parser_type * config = config_alloc();
  config_content_type * content = site_config_alloc_content(config);

  ert_workflow_list_init(workflow_list, content);

  config_free(config);
  config_content_free(content);

  return workflow_list;
}

ert_workflow_list_type * ert_workflow_list_alloc_load(
        const subst_list_type * context,
        const char * user_config_file) {

  config_parser_type * config_parser = config_alloc();
  config_content_type * config_content = NULL;
  if(user_config_file)
    config_content = model_config_alloc_content(user_config_file, config_parser);

  ert_workflow_list_type * workflow_list = ert_workflow_list_alloc(context, config_content);

  config_free(config_parser);
  config_content_free(config_content);

  return workflow_list;
}

ert_workflow_list_type * ert_workflow_list_alloc(
        const subst_list_type * context,
        const config_content_type * config_content) {

  ert_workflow_list_type * workflow_list = ert_workflow_list_alloc_load_site_config(context);

  if(config_content)
    ert_workflow_list_init(workflow_list, config_content);

  return workflow_list;
}

ert_workflow_list_type  *  ert_workflow_list_alloc_full(const subst_list_type * context,
                                                    workflow_joblist_type * workflow_joblist) {
  ert_workflow_list_type * workflow_list = ert_workflow_list_alloc_empty(context);
  workflow_list->joblist = workflow_joblist;
  workflow_list->context = context;

  return workflow_list;

}

UTIL_IS_INSTANCE_FUNCTION( ert_workflow_list , ERT_WORKFLOW_LIST_TYPE_ID )

void ert_workflow_list_set_verbose( ert_workflow_list_type * workflow_list , bool verbose) {
  workflow_list->verbose = verbose;
}


const subst_list_type * ert_workflow_list_get_context(const ert_workflow_list_type * workflow_list) {
    return workflow_list->context;
}

void ert_workflow_list_free( ert_workflow_list_type * workflow_list ) {
  hash_free( workflow_list->workflows );
  hash_free( workflow_list->alias_map );
  workflow_joblist_free( workflow_list->joblist );
  free( workflow_list );
}



workflow_type * ert_workflow_list_add_workflow( ert_workflow_list_type * workflow_list , const char * workflow_file , const char * workflow_name) {
  if (util_file_exists( workflow_file )) {
    workflow_type * workflow = workflow_alloc( workflow_file , workflow_list->joblist );
    char * name;

    if (workflow_name == NULL)
      util_alloc_file_components( workflow_file , NULL , &name , NULL );
    else
      name = (char *) workflow_name;


    hash_insert_hash_owned_ref( workflow_list->workflows , name , workflow , workflow_free__);
    if (hash_has_key( workflow_list->alias_map , name))
      hash_del( workflow_list->alias_map , name);

    if (workflow_name == NULL)
      free( name );

    return workflow;
  } else
    return NULL;
}



void ert_workflow_list_add_alias( ert_workflow_list_type * workflow_list , const char * real_name , const char * alias) {
  if (!util_string_equal( real_name , alias))
    hash_insert_ref( workflow_list->alias_map , alias , real_name );
}


void ert_workflow_list_add_job( ert_workflow_list_type * workflow_list , const char * job_name , const char * config_file ) {
  char * name = (char *) job_name;

  if (job_name == NULL)
    util_alloc_file_components( config_file , NULL , &name , NULL );

  if (!workflow_joblist_add_job_from_file( workflow_list->joblist , name , config_file ))
    fprintf(stderr,"** Warning: failed to add workflow job:%s from:%s \n", name , config_file );

  if (job_name == NULL)
    free(name);
}


bool ert_workflow_list_has_job( const ert_workflow_list_type * workflow_list , const char * job_name) {
  return workflow_joblist_has_job( workflow_list->joblist , job_name );
}


const workflow_job_type * ert_workflow_list_get_job( const ert_workflow_list_type * workflow_list , const char * job_name) {
    return workflow_joblist_get_job(workflow_list->joblist, job_name);
}


static char * ert_workflow_list_alloc_name( const char * path , const char * root_name ) {
  char * full_path = util_alloc_sprintf( "%s%s%s" , path , UTIL_PATH_SEP_STRING , root_name);

  if (util_is_file( full_path ))
    return full_path;
  else
    free( full_path );

  return NULL;
}


void ert_workflow_list_add_jobs_in_directory( ert_workflow_list_type * workflow_list , const char * path ) {
  DIR * dirH = opendir( path );
  std::set<std::string> names;
  if (!dirH) {
    fprintf(stderr, "** Warning: failed to open workflow/jobs directory: %s\n", path);
    return;
  }
  while (true) {
    struct dirent * entry = readdir( dirH );
    if (entry == NULL)
      break;

    if ((strcmp(entry->d_name , ".") == 0) || (strcmp(entry->d_name , "..") == 0))
      continue;

    char * root_name = entry->d_name;
    if (names.count(root_name))
      continue;

    char * full_path = ert_workflow_list_alloc_name( path , root_name );
    if (!full_path)
      continue;

    names.insert(root_name);
    res_log_finfo("Adding workflow job:%s", full_path);
    ert_workflow_list_add_job( workflow_list , root_name , full_path );
    free( full_path );
  }
  closedir( dirH );
}


stringlist_type * ert_workflow_list_get_job_names(const ert_workflow_list_type * workflow_list) {
    return  workflow_joblist_get_job_names(workflow_list->joblist);
}


static void ert_workflow_list_init(ert_workflow_list_type * workflow_list , const config_content_type * config) {
  /* Adding jobs */
  {
    if (config_content_has_item( config , WORKFLOW_JOB_DIRECTORY_KEY)) {
      const config_content_item_type * jobpath_item = config_content_get_item( config , WORKFLOW_JOB_DIRECTORY_KEY);
      for (int i=0; i < config_content_item_get_size( jobpath_item ); i++) {
        config_content_node_type * path_node = config_content_item_iget_node( jobpath_item , i );

        for (int j=0; j < config_content_node_get_size( path_node ); j++)
          ert_workflow_list_add_jobs_in_directory( workflow_list , config_content_node_iget_as_abspath( path_node , j ) );
      }
    }
  }

  {
    if (config_content_has_item( config , LOAD_WORKFLOW_JOB_KEY)) {
      const config_content_item_type * job_item = config_content_get_item( config , LOAD_WORKFLOW_JOB_KEY);
      for (int i=0; i < config_content_item_get_size( job_item ); i++) {
        config_content_node_type * job_node = config_content_item_iget_node( job_item , i );
        const char * config_file = config_content_node_iget_as_path( job_node , 0 );
        const char * job_name = config_content_node_safe_iget( job_node , 1 );
        ert_workflow_list_add_job( workflow_list , job_name , config_file);
      }
    }
  }


  /* Adding workflows */
  {
    if (config_content_has_item( config , LOAD_WORKFLOW_KEY)) {
      const config_content_item_type * workflow_item = config_content_get_item( config , LOAD_WORKFLOW_KEY);
      for (int i=0; i < config_content_item_get_size( workflow_item ); i++) {
        config_content_node_type * workflow_node = config_content_item_iget_node(workflow_item, i);

        const char * workflow_file = config_content_node_iget_as_abspath(workflow_node, 0);
        const char * workflow_name = config_content_node_safe_iget( workflow_node , 1 );

        ert_workflow_list_add_workflow( workflow_list , workflow_file , workflow_name );
      }
    }
  }
}


void ert_workflow_list_add_config_items( config_parser_type * config ) {
  config_schema_item_type * item = config_add_schema_item( config , WORKFLOW_JOB_DIRECTORY_KEY , false  );
  config_schema_item_set_argc_minmax(item , 1 , 1 );
  config_schema_item_iset_type( item , 0 , CONFIG_PATH );

  item = config_add_schema_item( config , LOAD_WORKFLOW_KEY , false  );
  config_schema_item_set_argc_minmax(item , 1 , 2 );
  config_schema_item_iset_type( item , 0 , CONFIG_EXISTING_PATH );

  item = config_add_schema_item( config , LOAD_WORKFLOW_JOB_KEY , false  );
  config_schema_item_set_argc_minmax(item , 1 , 2 );
  config_schema_item_iset_type( item , 0 , CONFIG_EXISTING_PATH );
}



workflow_type *  ert_workflow_list_get_workflow(ert_workflow_list_type * workflow_list , const char * workflow_name ) {
  const char * lookup_name = workflow_name;

  if (hash_has_key( workflow_list->alias_map , workflow_name))
    lookup_name = (const char * ) hash_get( workflow_list->alias_map , workflow_name );

  return (workflow_type * ) hash_get( workflow_list->workflows , lookup_name );
}

bool  ert_workflow_list_has_workflow(ert_workflow_list_type * workflow_list , const char * workflow_name ) {
  return
    hash_has_key( workflow_list->workflows , workflow_name ) ||
    hash_has_key( workflow_list->alias_map , workflow_name);
}


bool ert_workflow_list_run_workflow__(ert_workflow_list_type * workflow_list, workflow_type * workflow, bool verbose , void * self) {
  bool runOK = workflow_run( workflow, self , verbose , workflow_list->context);
  if (runOK)
    workflow_list->last_error = NULL;
  else
    workflow_list->last_error = workflow_get_last_error( workflow );

  return runOK;
}


bool ert_workflow_list_run_workflow_blocking(ert_workflow_list_type * workflow_list  , const char * workflow_name , void * self) {
  workflow_type * workflow = ert_workflow_list_get_workflow( workflow_list , workflow_name );
  bool result = ert_workflow_list_run_workflow__( workflow_list, workflow , workflow_list->verbose , self);
  return result;
}


bool ert_workflow_list_run_workflow(ert_workflow_list_type * workflow_list, const char * workflow_name , void * self) {
  workflow_type * workflow = ert_workflow_list_get_workflow( workflow_list , workflow_name );
  return ert_workflow_list_run_workflow__( workflow_list, workflow , workflow_list->verbose , self);
}


/*****************************************************************/

stringlist_type * ert_workflow_list_alloc_namelist( ert_workflow_list_type * workflow_list ) {
  return hash_alloc_stringlist( workflow_list->workflows );
}


const config_error_type * ert_workflow_list_get_last_error( const ert_workflow_list_type * workflow_list) {
  return workflow_list->last_error;
}


int ert_workflow_list_get_size( const ert_workflow_list_type * workflow_list) {
  return hash_get_size( workflow_list->workflows ) + hash_get_size( workflow_list->alias_map);
}
