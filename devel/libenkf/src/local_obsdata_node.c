/*
   Copyright (C) 2013  Statoil ASA, Norway.

   The file 'local_obsdata_node.c'

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

#include <ert/util/type_macros.h>
#include <ert/util/util.h>
#include <ert/util/int_vector.h>

#include <ert/enkf/local_obsdata_node.h>

#define LOCAL_OBSDATA_NODE_TYPE_ID 84441309

struct local_obsdata_node_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                * obs_key;
  active_list_type    * active_list;
  int_vector_type     * tstep_list;
};



UTIL_IS_INSTANCE_FUNCTION( local_obsdata_node , LOCAL_OBSDATA_NODE_TYPE_ID )

static UTIL_SAFE_CAST_FUNCTION( local_obsdata_node , LOCAL_OBSDATA_NODE_TYPE_ID )

local_obsdata_node_type * local_obsdata_node_alloc( const char * obs_key ) {
  local_obsdata_node_type * node = util_malloc( sizeof * node );
  UTIL_TYPE_ID_INIT( node , LOCAL_OBSDATA_NODE_TYPE_ID );
  node->obs_key = util_alloc_string_copy( obs_key );
  node->active_list = active_list_alloc( );
  node->tstep_list = int_vector_alloc(0,0);
  return node;
}


void local_obsdata_node_copy_active_list( local_obsdata_node_type * node , const active_list_type * active_list) {
  active_list_copy( node->active_list , active_list );
}


const char * local_obsdata_node_get_key( const local_obsdata_node_type * node ) {
  return node->obs_key;
}



void local_obsdata_node_free( local_obsdata_node_type * node ) {
  active_list_free( node->active_list );
  int_vector_free( node->tstep_list );
  free( node->obs_key );
  free( node );
}



void local_obsdata_node_free__( void * arg ) {
  local_obsdata_node_type * node = local_obsdata_node_safe_cast( arg );
  local_obsdata_node_free( node );
}


active_list_type * local_obsdata_node_get_active_list( const local_obsdata_node_type * node ) {
  return node->active_list;
}


const int_vector_type * local_obsdata_node_get_tstep_list( const local_obsdata_node_type * node) {
  return node->tstep_list;
}

bool local_obsdata_node_has_tstep( const local_obsdata_node_type * node , int tstep) {
  if (int_vector_index_sorted( node->tstep_list , tstep) == -1) 
    return false;
  else
    return true;
}


void local_obsdata_node_add_tstep( local_obsdata_node_type * node, int tstep) {
  if (!local_obsdata_node_has_tstep( node , tstep)) {
    if (int_vector_size( node->tstep_list )) {
      int last = int_vector_get_last( node->tstep_list );
      int_vector_append( node->tstep_list , tstep ); 
      if (tstep < last)
        int_vector_sort( node->tstep_list); 
    } else
      int_vector_append( node->tstep_list , tstep ); 
  }  
}




void local_obsdata_node_add_range( local_obsdata_node_type * node, int step1 , int step2) {
  int tstep;
  for (tstep = step1; tstep <= step2; tstep++)
    local_obsdata_node_add_tstep( node , tstep );
}
