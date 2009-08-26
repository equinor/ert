#include <stdlib.h>
#include <stdio.h>
#include <vector.h>
#include <util.h>
#include <local_ministep.h>
#include <local_updatestep.h>
#include <local_config.h>


/**
   This file implements the top level object in the system keeping
   track of active/inactive parameters and observations. The system is
   based on three levels.

        1. local_config_type - this implementation

        2. local_updatestep_type - what should be updated and which
           observations to use at one report step.
  
	3. local_ministep_type - what should be updated and which
           observations to use at one enkf update.
	   
*/



struct local_config_struct {
  vector_type           * updatestep;            /* This is an indexed vector with (pointers to) local_reportsstep instances. */
  local_updatestep_type * default_updatestep;    /* A default report step returned if no particular report step has been installed for this time index. */
  hash_type 		* updatestep_storage;    /* These two hash tables are the 'holding area' for the local_updatestep */
  hash_type 		* ministep_storage;      /* and local_ministep instances. */
};



/**
   Observe that history_length is *INCLUSIVE* 
*/
local_config_type * local_config_alloc( int history_length ) {
  local_config_type * local_config = util_malloc( sizeof * local_config , __func__);

  local_config->default_updatestep = NULL;
  local_config->updatestep_storage  = hash_alloc();
  local_config->ministep_storage    = hash_alloc();
  local_config->updatestep          = vector_alloc_new();
  {
    int report;
    for (report=0; report <= history_length; report++)
      vector_append_ref( local_config->updatestep , NULL );
  }
  
  return local_config;
}


void local_config_free(local_config_type * local_config) {
  vector_free( local_config->updatestep );
  hash_free( local_config->updatestep_storage );
  hash_free( local_config->ministep_storage);
  free( local_config );
}



/**
   Actual report step must have been installed in the
   updatestep_storage with local_config_alloc_updatestep() first.
*/

void local_config_set_default_updatestep( local_config_type * local_config , const char * default_key) {
  local_updatestep_type * default_updatestep = hash_get( local_config->updatestep_storage , default_key );
  local_config->default_updatestep = default_updatestep;
}


/**
   Instances of local_updatestep and local_ministep are allocated from
   the local_config object, and then subsequently manipulated from the calling scope.
*/

local_updatestep_type * local_config_alloc_updatestep( local_config_type * local_config , const char * key ) {
  local_updatestep_type * updatestep = local_updatestep_alloc( key );
  hash_insert_hash_owned_ref( local_config->updatestep_storage , key , updatestep , local_updatestep_free__);
  return updatestep;
}


local_ministep_type * local_config_alloc_ministep( local_config_type * local_config , const char * key ) {
  local_ministep_type * ministep = local_ministep_alloc( key );
  hash_insert_hash_owned_ref( local_config->ministep_storage , key , ministep , local_ministep_free__);
  return ministep;
}


local_ministep_type * local_config_alloc_ministep_copy( local_config_type * local_config , const char * src_key , const char * new_key) {
  local_ministep_type * src_step = hash_get( local_config->ministep_storage , src_key );
  local_ministep_type * new_step = local_ministep_alloc_copy( src_step , new_key );
  hash_insert_hash_owned_ref( local_config->ministep_storage , new_key , new_step , local_ministep_free__);
  return new_step;
}



const local_updatestep_type * local_config_iget_updatestep( const local_config_type * local_config , int index) {
  const local_updatestep_type * updatestep = vector_iget_const( local_config->updatestep , index );
  if (updatestep == NULL) 
    /* No particular report step has been installed for this index,
       revert to the default. */
    updatestep = local_config->default_updatestep;
  
  if (updatestep == NULL) 
    util_exit("%s: fatal error. No report step information for step:%d - and no default \n",__func__ , index);
    
  return updatestep;
}



/**
   This will 'install' the updatestep instance identified with 'key'
   for report steps [step1,step2]. Observe that the report step must
   have been allocated with 'local_config_alloc_updatestep()' first.
*/


void local_config_set_updatestep(local_config_type * local_config, int step1 , int step2 , const char * key) {
  local_updatestep_type * updatestep = hash_get( local_config->updatestep_storage , key );
  int step;
  
  for ( step = step1; step < util_int_min(step2 + 1 , vector_get_size( local_config->updatestep )); step++)
    vector_insert_ref(local_config->updatestep , step , updatestep );
  
}
