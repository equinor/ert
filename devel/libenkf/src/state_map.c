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
  int_vector_type  * state;
  pthread_rwlock_t   rw_lock;
};


UTIL_IS_INSTANCE_FUNCTION( state_map , STATE_MAP_TYPE_ID )


state_map_type * state_map_alloc( ) {
  state_map_type * map = util_malloc( sizeof * map );
  UTIL_TYPE_ID_INIT( map , STATE_MAP_TYPE_ID );
  map->state = int_vector_alloc( 0 , STATE_UNDEFINED );
  pthread_rwlock_init( &map->rw_lock , NULL);
  return map;
}


state_map_type * state_map_fread_alloc( const char * filename ) {
  state_map_type * map = state_map_alloc();
  if (util_file_exists( filename )) {
    FILE * stream = util_fopen( filename , "r");
    int_vector_fread( map->state , stream );
    fclose( stream );
  } 
  return map;
}

state_map_type * state_map_alloc_copy( state_map_type * map ) {
  state_map_type * copy = state_map_alloc();
  pthread_rwlock_rdlock( &map->rw_lock );
  {
    int_vector_memcpy( copy->state , map->state );
  }
  pthread_rwlock_unlock( &map->rw_lock );
  return copy;
}

void state_map_free( state_map_type * map ) {
  free( map );
}


int state_map_get_size( state_map_type * map) {
  int size;
  pthread_rwlock_rdlock( &map->rw_lock );
  {
    size = int_vector_size( map->state );
  }
  pthread_rwlock_unlock( &map->rw_lock );
  return size;
}


bool state_map_equal( state_map_type * map1 , state_map_type * map2) {
  bool equal = true;
  pthread_rwlock_rdlock( &map1->rw_lock );
  pthread_rwlock_rdlock( &map2->rw_lock );
  {
    int size1 = int_vector_size( map1->state);
    if (size1 != int_vector_size( map2->state))
      equal = false;

    if (equal) 
      equal = int_vector_equal( map1->state , map2->state );
  }
  pthread_rwlock_unlock( &map1->rw_lock );
  pthread_rwlock_unlock( &map2->rw_lock );
  return equal;
}


realisation_state_enum state_map_iget( state_map_type * map , int index) {
  realisation_state_enum state;
  pthread_rwlock_rdlock( &map->rw_lock );
  {
    state = int_vector_safe_iget( map->state , index );
  }
  pthread_rwlock_unlock( &map->rw_lock );
  return state;
}

void state_map_iset( state_map_type * map ,int index , realisation_state_enum state) {
  pthread_rwlock_wrlock( &map->rw_lock );
  {
    int_vector_iset( map->state , index , state );
  }
  pthread_rwlock_unlock( &map->rw_lock );
}

void state_map_update_undefined( state_map_type * map , int index , realisation_state_enum new_state) {
  realisation_state_enum current_state = state_map_iget( map , index );
  if (current_state == STATE_UNDEFINED)
    state_map_iset( map , index , new_state );
}


void state_map_fwrite( state_map_type * map , const char * filename) {
  pthread_rwlock_rdlock( &map->rw_lock );
  {
    FILE * stream = util_mkdir_fopen( filename , "w");
    if (stream) {
      int_vector_fwrite( map->state , stream );
      fclose( stream );
    } else
      util_abort("%s: failed to open:%s for writing \n",__func__ , filename );
  }
  pthread_rwlock_unlock( &map->rw_lock );
}



void state_map_fread( state_map_type * map , const char * filename) {
  pthread_rwlock_wrlock( &map->rw_lock );
  {
    if (util_file_exists( filename )) {
      FILE * stream = util_fopen( filename , "r");
      if (stream) {
        int_vector_fread( map->state , stream );
        fclose( stream );
      } else
        util_abort("%s: failed to open:%s for reading \n",__func__ , filename );
    } else
      int_vector_reset( map->state );
  }
  pthread_rwlock_unlock( &map->rw_lock );
}
