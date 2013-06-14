/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'local_obsdata.c'
    
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
#include <ert/util/type_macros.h>
#include <ert/util/vector.h>

#include <ert/enkf/local_obsdata.h>


#define LOCAL_OBSDATA_TYPE_ID 86331309

struct local_obsdata_struct {
  UTIL_TYPE_ID_DECLARATION;
  vector_type * obs_nodes;
};



UTIL_IS_INSTANCE_FUNCTION( local_obsdata  , LOCAL_OBSDATA_TYPE_ID )

local_obsdata_type * local_obsdata_alloc( ) {
  local_obsdata_type * data = util_malloc( sizeof * data );
  UTIL_TYPE_ID_INIT( data , LOCAL_OBSDATA_TYPE_ID );
  data->obs_nodes = vector_alloc_new();
  return data;
}



void local_obsdata_free( local_obsdata_type * data ) {
  vector_free( data->obs_nodes );
  free( data );
}


int local_obsdata_get_size( const local_obsdata_type * data ) {
  return vector_get_size( data->obs_nodes );
}


/*
  The @data instance will assume ownership of the node; i.e. calling
  scope should NOT call local_obsdata_node_free().
*/

void local_obsdata_add_node( local_obsdata_type * data , local_obsdata_node_type * node ) {
  vector_append_owned_ref( data->obs_nodes , node , local_obsdata_node_free__ );
}


const local_obsdata_node_type * local_obsdata_iget( const local_obsdata_type * data , int index) {
  return vector_iget_const( data->obs_nodes , index );
}
