/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
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

#include <hash.h>
#include <stringlist.h>
#include <util.h>

#include <config.h>
#include <config_schema_item.h>

#include <workflow.h>
#include <workflow_job.h>
#include <workflow_joblist.h>

#include <ert_workflow_list.h>
#include <config_keys.h>

struct ert_workflow_list_struct {
  stringlist_type       * path_list;
  hash_type             * workflows;
  workflow_joblist_type * joblist;
};



ert_workflow_list_type * ert_workflow_list_alloc() {
  ert_workflow_list_type * workflow_list = util_malloc( sizeof * workflow_list );
  workflow_list->path_list = stringlist_alloc_new();
  workflow_list->workflows = hash_alloc();
  workflow_list->joblist   = workflow_joblist_alloc();
  return workflow_list;
}



void ert_workflow_list_free( ert_workflow_list_type * workflow_list ) {
  hash_free( workflow_list->workflows );
  stringlist_free( workflow_list->path_list );
  workflow_joblist_free( workflow_list->joblist );
  free( workflow_list );
}



void ert_workflow_list_add_workflow( ert_workflow_list_type * workflow_list , const char * workflow_file , const char * workflow_name) {
  workflow_type * workflow = workflow_alloc( workflow_file , workflow_list->joblist );
  char * name = workflow_name;
  if (workflow_name == NULL) 
    util_alloc_file_components( workflow_file , NULL , &name , NULL );

  hash_insert_hash_owned_ref( workflow_list->workflows , name , workflow , workflow_free__);
  
  if (workflow_name == NULL) 
    free( name );
}



void ert_workflow_list_add_job( ert_workflow_list_type * workflow_list , const char * job_name , const char * config_file ) {
  if (!workflow_joblist_add_job_from_file( workflow_list->joblist , job_name , config_file )) {
    fprintf(stderr,"** Warning: failed to add workflow job:%s from:%s \n",job_name , config_file );
  }
}



void ert_workflow_list_add_jobs_in_directory( ert_workflow_list_type * workflow_list , const char * path) {
  DIR * dirH = opendir( path );
  while (true) {
    struct dirent * entry = readdir( dirH );
    if (entry != NULL) {
      if ((strcmp(entry->d_name , ".") != 0) && (strcmp(entry->d_name , "..") != 0)) {
        char * full_path = util_alloc_filename( path , entry->d_name , NULL );

        if (util_is_file( full_path ))
          ert_workflow_list_add_job( workflow_list , entry->d_name , full_path );
        
        free( full_path );
      }
    } else 
      break;
  }
  closedir( dirH );
}


void ert_workflow_list_init( ert_workflow_list_type * workflow_list , config_type * config ) {
  /* Adding jobs */
  for (int i=0; i < config_get_occurences( config , WORKFLOW_JOB_DIRECTORY_KEY ); i++) {
    const stringlist_type * path_list = config_iget_stringlist_ref( config , WORKFLOW_JOB_DIRECTORY_KEY , i);
    for (int j=0; j < stringlist_get_size( path_list ); j++) 
      ert_workflow_list_add_jobs_in_directory( workflow_list , stringlist_iget( path_list , j ));
  }

  /* Adding workflows */
  for (int i=0; i < config_get_occurences( config , LOAD_WORKFLOW_KEY ); i++) {
    const stringlist_type * workflow = config_iget_stringlist_ref( config , LOAD_WORKFLOW_KEY , i);
    const char * workflow_file = stringlist_iget( workflow , 0 );
    char * workflow_name = stringlist_safe_iget( workflow , 1 );
    
    ert_workflow_list_add_workflow( workflow_list , workflow_file , workflow_name );
  }
}


void ert_workflow_list_update_config( config_type * config ) {
  config_schema_item_type * item = config_add_schema_item( config , WORKFLOW_JOB_DIRECTORY_KEY , false  );
  config_schema_item_set_argc_minmax(item , 1 , 1 , 0 , NULL);

  item = config_add_schema_item( config , LOAD_WORKFLOW_KEY , false  );
  config_schema_item_set_argc_minmax(item , 1 , 2 , 0 , NULL);
}



workflow_type *  workflow_list_get_workflow(ert_workflow_list_type * workflow_list , const char * workflow_name ) {
  return hash_get( workflow_list->workflows , workflow_name );
}

void ert_workflow_list_run_workflow(ert_workflow_list_type * workflow_list  , const char * workflow_name , void * self) {
  workflow_type * workflow = workflow_list_get_workflow( workflow_list , workflow_name );
  workflow_run( workflow , self );
}
