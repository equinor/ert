/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'local_ministep.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <string.h>

#include <ert/util/hash.h>

#include <ert/enkf/local_config.hpp>

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
  UTIL_TYPE_ID_DECLARATION;
  char              * name;             /* A name used for this ministep - string is also used as key in a hash table holding this instance. */
  hash_type         * datasets;         /* A hash table of local_dataset_type instances - indexed by the name of the datasets. */
  local_obsdata_type * observations;
  analysis_module_type * analysis_module;
  obs_data_type * obs_data;
};


/**
   Observe there is no link between the instances here and the real
   observations/nodes (apart from the key in the hash).
*/

UTIL_SAFE_CAST_FUNCTION(local_ministep , LOCAL_MINISTEP_TYPE_ID)
UTIL_IS_INSTANCE_FUNCTION(local_ministep , LOCAL_MINISTEP_TYPE_ID)

local_ministep_type * local_ministep_alloc(const char * name, analysis_module_type* analysis_module) {
  local_ministep_type * ministep = (local_ministep_type *)util_malloc( sizeof * ministep );

  ministep->name         = util_alloc_string_copy( name );

  const char* obsdata_name = "OBSDATA_";
  char* result = (char *) util_malloc(strlen(obsdata_name)+strlen(name)+1);
  strcpy(result, obsdata_name);
  strcat(result, name);
  ministep->observations = local_obsdata_alloc(result);
  free(result);


  ministep->datasets     = hash_alloc();
  ministep->analysis_module = analysis_module;
  ministep->obs_data = NULL;
  UTIL_TYPE_ID_INIT( ministep , LOCAL_MINISTEP_TYPE_ID);

  return ministep;
}

void local_ministep_free(local_ministep_type * ministep) {
  free(ministep->name);
  hash_free( ministep->datasets );
  local_obsdata_free(ministep->observations);
  if ( ministep->obs_data != NULL )
    obs_data_free(ministep->obs_data);
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



void local_ministep_add_dataset( local_ministep_type * ministep , const local_dataset_type * dataset) {
  hash_insert_ref( ministep->datasets , local_dataset_get_name( dataset ) , dataset );
}

void local_ministep_add_obsdata( local_ministep_type * ministep , local_obsdata_type * obsdata) {
  if (ministep->observations == NULL)
    ministep->observations = obsdata;
  else { // Add nodes from input observations to existing observations
    int iobs;
    for (iobs = 0; iobs < local_obsdata_get_size( obsdata ); iobs++) {
      local_obsdata_node_type * obs_node = local_obsdata_iget( obsdata , iobs );
      local_obsdata_node_type * new_node = local_obsdata_node_alloc_copy(obs_node);
      local_ministep_add_obsdata_node(ministep, new_node);
    }
  }
}

void local_ministep_add_obs_data( local_ministep_type * ministep , obs_data_type * obs_data) {
  if (ministep->obs_data != NULL){
    obs_data_free(ministep->obs_data);
    ministep->obs_data = NULL;
  }
  ministep->obs_data = obs_data;
}

void local_ministep_add_obsdata_node( local_ministep_type * ministep , local_obsdata_node_type * obsdatanode) {
  local_obsdata_type * obsdata = local_ministep_get_obsdata(ministep);
  local_obsdata_add_node(obsdata, obsdatanode);
}

bool local_ministep_has_dataset( const local_ministep_type * ministep, const char * dataset_name) {
  return hash_has_key( ministep->datasets, dataset_name );
}

int local_ministep_get_num_dataset( const local_ministep_type * ministep ) {
  return hash_get_size( ministep->datasets );
}

local_dataset_type * local_ministep_get_dataset( const local_ministep_type * ministep, const char * dataset_name) {
  return (local_dataset_type *) hash_get( ministep->datasets, dataset_name ); // CXX_CAST_ERROR
}

local_obsdata_type * local_ministep_get_obsdata( const local_ministep_type * ministep ) {
  return ministep->observations;
}

obs_data_type * local_ministep_get_obs_data( const local_ministep_type * ministep ) {
  return ministep->obs_data;
}

const char * local_ministep_get_name( const local_ministep_type * ministep ) {
  return ministep->name;
}

/*****************************************************************/

hash_iter_type * local_ministep_alloc_dataset_iter( const local_ministep_type * ministep ) {
  return hash_iter_alloc( ministep->datasets );
}


bool local_ministep_has_analysis_module( const local_ministep_type * ministep){
  return ministep->analysis_module != NULL;
}

analysis_module_type* local_ministep_get_analysis_module( const local_ministep_type * ministep ){
  return ministep->analysis_module;
}

void local_ministep_summary_fprintf( const local_ministep_type * ministep , FILE * stream) {

  fprintf(stream , "MINISTEP:%s,", ministep->name);

  {
    /* Dumping all the DATASET instances. */
    {
     hash_iter_type * dataset_iter = hash_iter_alloc( ministep->datasets );
     while (!hash_iter_is_complete( dataset_iter )) {
       const local_dataset_type * dataset = (const local_dataset_type *) hash_iter_get_next_value( dataset_iter );
       local_dataset_summary_fprintf(dataset, stream);
     }
     hash_iter_free( dataset_iter );
    }

    /* Only one OBSDATA */
   local_obsdata_type * obsdata = local_ministep_get_obsdata(ministep);
   local_obsdata_summary_fprintf( obsdata , stream);
   fprintf(stream, "\n");
  }
}
