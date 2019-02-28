/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'config_path_stack.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_CONFIG_PATH_STACK_H
#define ERT_CONFIG_PATH_STACK_H

#include <ert/config/config_path_elm.hpp>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct config_path_stack_struct config_path_stack_type;

void                      config_path_stack_free( config_path_stack_type * path_stack );
config_path_stack_type *  config_path_stack_alloc( );
void                      config_path_stack_append( config_path_stack_type * path_stack , config_path_elm_type * path_elm);
int                       config_path_stack_size( const config_path_stack_type * path_stack );
const                     config_path_elm_type * config_path_stack_get_last( const config_path_stack_type * path_stack );
void                      config_path_stack_pop( config_path_stack_type * path_stack );

#ifdef __cplusplus
}
#endif
#endif
