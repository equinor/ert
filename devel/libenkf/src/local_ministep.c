#include <util.h>
#include <stdlib.h>
#include <string.h>
#include <active_list.h>
#include <hash.h>
#include <enkf_macros.h>
#include <local_ministep.h>

/**
   This file implements a 'ministep' configuration for active /
   inactive observations and parameters for ONE enkf update. Observe
   that the updating at one report step can consist of several
   socalled ministeps, i.e. first the northern part of the field with
   the relevant observations, and then the southern part.

   The implementation, in local_ministep_type, is quite simple, it
   only contains the keys for the observations and nodes, with an
   accompanying pointer to an active_list instance which denotes the
   active indices. Observe that this implementation offers no access
   to the internals of the underlying enkf_node / obs_node objects.
*/


#define LOCAL_MINISTEP_TYPE_ID 661066  

struct local_ministep_struct {
  int         __type_id;
  char      * name;             /* A name used for this ministep - string is also used as key in a hash table holding this instance. */
  hash_type * observations;     /* A hash table indexed by observation keys - each element is an active_list instance. */
  hash_type * nodes;            /* A hash table indexed by node keys - each element is an active_list instance. */
};


/**
   Observe there is no link between the instances here and the real
   observations/nodes (apart from the key in the hash).
*/

SAFE_CAST(local_ministep , LOCAL_MINISTEP_TYPE_ID)


local_ministep_type * local_ministep_alloc(const char * name) {
  local_ministep_type * ministep = util_malloc( sizeof * ministep , __func__ );

  ministep->name         = util_alloc_string_copy( name );
  ministep->observations = hash_alloc();
  ministep->nodes        = hash_alloc();
  ministep->__type_id    = LOCAL_MINISTEP_TYPE_ID;
  
  return ministep;
}


/**
   This is a deep copy.
*/
local_ministep_type * local_ministep_alloc_copy( const local_ministep_type * src , const char * name) {
  local_ministep_type * new = local_ministep_alloc( name );
  {
    hash_iter_type * obs_iter = hash_iter_alloc( src->observations );
    while (!hash_iter_is_complete( obs_iter )) {
      const char * obs_key = hash_iter_get_next_key( obs_iter );
      active_list_type * active_list_copy = active_list_alloc_copy( hash_get( src->observations , obs_key) );
      hash_insert_hash_owned_ref( new->observations , obs_key , active_list_copy , active_list_free__);
    }
  }

  {
    hash_iter_type * node_iter = hash_iter_alloc( src->nodes );
    while (!hash_iter_is_complete( node_iter )) {
      const char * node_key = hash_iter_get_next_key( node_iter );
      active_list_type * active_list_copy = active_list_alloc_copy( hash_get( src->nodes , node_key) );
      hash_insert_hash_owned_ref( new->nodes , node_key , active_list_copy , active_list_free__);
    }
  }
  
  return new;
}



void local_ministep_free(local_ministep_type * ministep) {
  free(ministep->name);
  hash_free( ministep->observations );
  hash_free( ministep-> nodes );
  free( ministep );
}


void local_ministep_free__(void * arg) {
  local_ministep_type * ministep = local_ministep_safe_cast( arg );
  local_ministep_free( ministep );
}




/**
   When adding observations and update nodes here observe the following:

    1. The thing will fail hard if you try to add a node/obs which is
       already in the hash table.

    2. The newly added elements will be assigned an active_list
       instance with mode ALL_ACTIVE.
*/   


void local_ministep_add_obs(local_ministep_type * ministep, const char * obs_key) {
  if (hash_has_key( ministep->observations , obs_key ))
    util_abort("%s: tried to add existing observation key:%s \n",__func__ , obs_key);

  hash_insert_hash_owned_ref( ministep->observations , obs_key , active_list_alloc( ALL_ACTIVE ) , active_list_free__);
}


void local_ministep_add_node(local_ministep_type * ministep, const char *node_key) {
  if (hash_has_key( ministep->nodes , node_key ))
    util_abort("%s: tried to add existing node key:%s \n",__func__ , node_key);

  hash_insert_hash_owned_ref( ministep->nodes , node_key , active_list_alloc( ALL_ACTIVE ) , active_list_free__);
}


void local_ministep_del_obs( local_ministep_type * ministep , const char * obs_key) {
  hash_del( ministep->observations , obs_key );
}


void local_ministep_del_node( local_ministep_type * ministep , const char * node_key) {
  hash_del( ministep->nodes , node_key );
}


void local_ministep_clear_observations( local_ministep_type * ministep) {
  hash_clear( ministep->observations );
}


void local_ministep_clear_nodes( local_ministep_type * ministep) {
  hash_clear( ministep->nodes );
}





active_list_type * local_ministep_get_obs_active_list(const local_ministep_type * ministep , const char * obs_key ) {
  return hash_get( ministep->observations , obs_key );  /* Fails hard if you do not have the key ... */
}


active_list_type * local_ministep_get_node_active_list(const local_ministep_type * ministep , const char * node_key ) {
  return hash_get( ministep->nodes , node_key );  /* Fails hard if you do not have the key ... */
}

const char * local_ministep_get_name( const local_ministep_type * ministep ) {
  return ministep->name;
}


/*****************************************************************/

hash_iter_type * local_ministep_alloc_node_iter(const local_ministep_type * ministep) {
  return hash_iter_alloc( ministep->nodes );
}

stringlist_type * local_ministep_alloc_node_keys(const local_ministep_type * ministep) {
  return hash_alloc_stringlist( ministep->nodes );
}


hash_iter_type * local_ministep_alloc_obs_iter(const local_ministep_type * ministep) {
  return hash_iter_alloc( ministep->observations );
}
