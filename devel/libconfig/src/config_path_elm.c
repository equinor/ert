/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'config_path_elm.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <string.h>
#include <stdio.h>

#include <type_macros.h>
#include <util.h>

#include <config_path_elm.h>

#define CONFIG_PATH_ELM_TYPE_ID  7100063

struct config_path_elm_struct {
  UTIL_TYPE_ID_DECLARATION;
  char * abs_path;     // This will always be absolute
  char * rel_path;     // This will be relative IFF the input path to the constructor is relative
};

static UTIL_SAFE_CAST_FUNCTION( config_path_elm , CONFIG_PATH_ELM_TYPE_ID )


config_path_elm_type * config_path_elm_alloc( const char * path)  {
  config_path_elm_type * path_elm = util_malloc( sizeof * path_elm );
  UTIL_TYPE_ID_INIT(path_elm , CONFIG_PATH_ELM_TYPE_ID);
  if (path == NULL) {
    path_elm->rel_path = NULL;
    path_elm->abs_path = util_alloc_cwd();
  } else {
    path_elm->rel_path = util_alloc_string_copy( path );
    if (util_is_abs_path( path ))
      path_elm->abs_path = util_alloc_string_copy( path );
    else
      path_elm->abs_path = util_alloc_abs_path( path );
  }

  return path_elm;
}


void config_path_elm_free( config_path_elm_type * path_elm ) {
  util_safe_free( path_elm->rel_path );
  util_safe_free( path_elm->abs_path );
  free( path_elm );
}



void config_path_elm_free__( void * arg ) {
  config_path_elm_type * path_elm = config_path_elm_safe_cast( arg );
  config_path_elm_free( path_elm );
}


const char * config_path_elm_get_path( const config_path_elm_type * path_elm ) {
  return path_elm->rel_path;
}


const char * config_path_elm_get_abspath( const config_path_elm_type * path_elm ) {
  return path_elm->abs_path;
}

char * config_path_elm_alloc_path(const config_path_elm_type * path_elm , const char * path) {
  return util_alloc_filename( path_elm->rel_path , path , NULL );
}


char * config_path_elm_alloc_abspath(const config_path_elm_type * path_elm , const char * path) {
  return util_alloc_filename( path_elm->abs_path , path , NULL );
}
