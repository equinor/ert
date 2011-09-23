/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'local_nodeset.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <util.h>
#include <stdlib.h>
#include <string.h>
#include <hash.h>
#include <enkf_macros.h>
#include <local_ministep.h>
#include <local_config.h>  
#include <active_list.h>
#include <local_nodeset.h>


#define LOCAL_NODESET_TYPE_ID  6615409

struct local_nodeset_struct {
  UTIL_TYPE_ID_DECLARATION;
  char      * name;
  hash_type * nodes;            /* A hash table indexed by node keys - each element is an active_list instance. */
};



UTIL_SAFE_CAST_FUNCTION(local_nodeset , LOCAL_NODESET_TYPE_ID)
UTIL_IS_INSTANCE_FUNCTION(local_nodeset , LOCAL_NODESET_TYPE_ID)


local_nodeset_type * local_nodeset_alloc( const char * name ) {
  local_nodeset_type * nodeset = util_malloc( sizeof * nodeset , __func__ );

  UTIL_TYPE_ID_INIT( nodeset , LOCAL_NODESET_TYPE_ID );
  nodeset->nodes = hash_alloc();
  nodeset->name  = util_alloc_string_copy( name );

  return nodeset;
}


void local_nodeset_free( local_nodeset_type * nodeset ) {
  util_safe_free(nodeset->name);
  hash_free( nodeset->nodes );
  free( nodeset );
}

void local_nodeset_free__( void * arg ) {
  local_nodeset_type * local_nodeset = local_nodeset_safe_cast( arg );
  local_nodeset_free( local_nodeset );
}

const char * local_nodeset_get_name( const local_nodeset_type * nodeset) {
  return nodeset->name;
}


void local_nodeset_add_node(local_nodeset_type * nodeset, const char *node_key) {
  if (hash_has_key( nodeset->nodes , node_key ))
    util_abort("%s: tried to add existing node key:%s \n",__func__ , node_key);

  hash_insert_hash_owned_ref( nodeset->nodes , node_key , active_list_alloc( ALL_ACTIVE ) , active_list_free__);
}


void local_nodeset_del_node( local_nodeset_type * nodeset , const char * node_key) {
  hash_del( nodeset->nodes , node_key );
}


void local_nodeset_clear( local_nodeset_type * nodeset) {
  hash_clear( nodeset->nodes );
}


active_list_type * local_nodeset_get_node_active_list(const local_nodeset_type * nodeset , const char * node_key ) {
  return hash_get( nodeset->nodes , node_key );  /* Fails hard if you do not have the key ... */
}

stringlist_type * local_nodeset_alloc_keys( const local_nodeset_type * nodeset ) {
  return hash_alloc_keylist( nodeset->nodes );
}


void local_nodeset_fprintf( const local_nodeset_type * nodeset , FILE * stream) {
  hash_iter_type * data_iter = hash_iter_alloc( nodeset->nodes );
  while (!hash_iter_is_complete( data_iter )) {
    const char * data_key          = hash_iter_get_next_key( data_iter );
    active_list_type * active_list = hash_get( nodeset->nodes , data_key );
    
    fprintf(stream , "%s %s %s\n", local_config_get_cmd_string( ADD_DATA ) , nodeset->name , data_key );
    active_list_fprintf( active_list , false , data_key , stream );
  }
  hash_iter_free( data_iter );
}



int local_nodeset_get_size( const local_nodeset_type * nodeset ) {
  return hash_get_size( nodeset->nodes );
}

