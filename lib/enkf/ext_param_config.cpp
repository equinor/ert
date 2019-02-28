/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'ext_param_config.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include <ert/util/util.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/ext_param_config.hpp>

#define EXT_PARAM_CONFIG_ID 97124451
struct ext_param_config_struct {
  UTIL_TYPE_ID_DECLARATION;
  char * key;
  stringlist_type * keys;
};

UTIL_SAFE_CAST_FUNCTION(ext_param_config, EXT_PARAM_CONFIG_ID)
UTIL_SAFE_CAST_FUNCTION_CONST(ext_param_config, EXT_PARAM_CONFIG_ID)

void ext_param_config_free( ext_param_config_type * config ) {
  free( config );
}

int ext_param_config_get_data_size( const ext_param_config_type * config ) {
  return stringlist_get_size( config->keys );
}


const char* ext_param_config_iget_key( const ext_param_config_type * config , int index) {
  return stringlist_iget( config->keys , index);
}

int ext_param_config_get_key_index( const ext_param_config_type * config , const char * key) {
  return stringlist_find_first( config->keys , key );
}


bool ext_param_config_has_key( const ext_param_config_type * config , const char * key) {
  return (ext_param_config_get_key_index(config, key) >= 0);
}


ext_param_config_type * ext_param_config_alloc( const char * key, const stringlist_type * keys) {
  if (stringlist_get_size( keys ) == 0)
    return NULL;

  if (!stringlist_unique( keys ))
    return NULL;

  ext_param_config_type * config = (ext_param_config_type *)util_malloc( sizeof * config );
  UTIL_TYPE_ID_INIT( config , EXT_PARAM_CONFIG_ID);
  config->key = util_alloc_string_copy( key );
  config->keys = stringlist_alloc_deep_copy( keys );
  return config;
}

VOID_FREE(ext_param_config)
VOID_GET_DATA_SIZE(ext_param)
