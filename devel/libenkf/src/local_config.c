#include <stdlib.h>
#include <stdio.h>
#include <vector.h>
#include <util.h>
#include <local_ministep.h>
#include <local_reportstep.h>
#include <local_config.h>


/**
   This file implements the top level object in the system keeping
   track of active/inactive parameters and observations. The system is
   based on three levels.

        1. local_config_type - this implementation

        2. local_reportstep_type - what should be updated and which
           observations to use at one report step.
  
	3. local_ministep_type - what should be updated and which
           observations to use at one enkf update.
	   
*/



struct local_config_struct {
  vector_type           * reportstep;            /* This is an indexed vector with (pointers to) local_reportsstep instances. */
  
  local_reportstep_type * default_reportstep;    /* A default report step returned if no particular report step has been installed for this time index. */
  
  hash_type 		* reportstep_storage;    /* These two hash tables are the 'holding area' for the local_reportstep */
  hash_type 		* ministep_storage;      /* and local_ministep instances. */
};



/**
   Observe that history_length is *INCLUSIVE* 
*/
local_config_type * local_config_alloc( int history_length ) {
  local_config_type * local_config = util_malloc( sizeof * local_config , __func__);

  local_config->default_reportstep = NULL;
  local_config->reportstep_storage  = hash_alloc();
  local_config->ministep_storage    = hash_alloc();
  local_config->reportstep          = vector_alloc_new();
  {
    int report;
    for (report=0; report <= history_length; report++)
      vector_append_ref( local_config->reportstep , NULL );
  }
  
  return local_config;
}


void local_config_free(local_config_type * local_config) {
  vector_free( local_config->reportstep );
  hash_free( local_config->reportstep_storage );
  hash_free( local_config->ministep_storage);
  free( local_config );
}


/**
   Actual report step must have been installed in the
   reportstep_storage with local_config_alloc_reportstep() first.
*/


void local_config_set_default_reportstep( local_config_type * local_config , const char * default_key) {
  local_reportstep_type * default_reportstep = hash_get( local_config->reportstep_storage , default_key );
  local_config->default_reportstep = default_reportstep;
}


/**
   Instances of local_reportstep and local_ministep are allocated from
   the local_config object, and then subsequently manipulated from the calling scope.
*/

local_reportstep_type * local_config_alloc_reportstep( local_config_type * local_config , const char * key ) {
  local_reportstep_type * reportstep = local_reportstep_alloc( key );
  hash_insert_hash_owned_ref( local_config->reportstep_storage , key , reportstep , local_reportstep_free__);
  return reportstep;
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



const local_reportstep_type * local_config_iget_reportstep( const local_config_type * local_config , int index) {
  const local_reportstep_type * reportstep = vector_iget_const( local_config->reportstep , index );
  if (reportstep == NULL) 
    /* No particular report step has been installed for this index,
       revert to the default. */
    reportstep = local_config->default_reportstep;
  
  if (reportstep == NULL) 
    util_exit("%s: fatal error. No report step information for step:%d - and no default \n",__func__ , index);
    
  return reportstep;
}



/**
   This will 'install' the reportstep instance identified with 'key'
   for report steps [step1,step2]. Observe that the report step must
   have been allocated with 'local_config_alloc_reportstep()' first.
*/


void local_config_set_reportstep(local_config_type * local_config, int step1 , int step2 , const char * key) {
  local_reportstep_type * reportstep = hash_get( local_config->reportstep_storage , key );
  int step;
  
  for ( step = step1; step < util_int_min(step2 + 1 , vector_get_size( local_config->reportstep )); step++)
    vector_insert_ref(local_config->reportstep , step , reportstep );
  
}
