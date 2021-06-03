/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'config_path_stack.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/type_macros.hpp>
#include <ert/util/vector.hpp>

#include <ert/config/config_path_elm.hpp>
#include <ert/config/config_path_stack.hpp>

#define CONFIG_PATH_STACK_TYPE_ID 86751520

struct config_path_stack_struct {
  UTIL_TYPE_ID_DECLARATION;
  vector_type * storage;
  vector_type * stack;
};



config_path_stack_type * config_path_stack_alloc( ) {
  config_path_stack_type * path_stack = (config_path_stack_type*)util_malloc( sizeof * path_stack);
  UTIL_TYPE_ID_INIT( path_stack , CONFIG_PATH_STACK_TYPE_ID );
  path_stack->storage  = vector_alloc_new();
  path_stack->stack    = vector_alloc_new();
  return path_stack;
}


void config_path_stack_free( config_path_stack_type * path_stack ) {
  vector_free( path_stack->storage );
  vector_free( path_stack->stack );
}


void config_path_stack_append( config_path_stack_type * path_stack , config_path_elm_type * path_elm) {
  vector_append_owned_ref( path_stack->storage , path_elm , config_path_elm_free__);
  vector_append_ref( path_stack->stack , path_elm );
}


int config_path_stack_size( const config_path_stack_type * path_stack ) {
  return vector_get_size( path_stack->stack );
}

const config_path_elm_type * config_path_stack_get_last( const config_path_stack_type * path_stack ) {
  return (const config_path_elm_type*)vector_get_last_const(path_stack->stack);
}


void config_path_stack_pop( config_path_stack_type * path_stack ) {
  vector_pop_back( path_stack->stack );
}
