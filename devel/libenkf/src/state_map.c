/*
   Copyright (C) 2013  Statoil ASA, Norway. 
   The file 'state_map.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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


#define  _GNU_SOURCE   /* Must define this to get access to pthread_rwlock_t */
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>

#include <ert/util/util.h>
#include <ert/util/int_vector.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_types.h>
#include <ert/enkf/state_map.h>


#define STATE_MAP_TYPE_ID 500672132

struct state_map_struct {
  UTIL_TYPE_ID_DECLARATION;
  int_vector_type * state;
};


UTIL_IS_INSTANCE_FUNCTION( state_map , STATE_MAP_TYPE_ID )


state_map_type * state_map_alloc( ) {
  state_map_type * map = util_malloc( sizeof * map );
  UTIL_TYPE_ID_INIT( map , STATE_MAP_TYPE_ID );
  map->state = int_vector_alloc( 0 , STATE_UNDEFINED );
  return map;
}


void state_map_free( state_map_type * map ) {
  free( map );
}


int state_map_get_size( const state_map_type * map) {
  return int_vector_size( map->state );
}
