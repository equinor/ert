/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'misfit_ensemble.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include <util.h>
#include <hash.h>
#include <vector.h>
#include <double_vector.h>
#include <msg.h>
#include <buffer.h>

#include <enkf_obs.h>
#include <enkf_fs.h>
#include <enkf_util.h>
#include <misfit_ensemble.h>
#include <misfit_ranking.h>

/**
   This file implements a type misfit_ensemble which is used to rank the
   different realization according to various criteria.

   The top level datastructure in this file is the misfit_ensemble, and
   that is the only exported datatype, but in addition there are the
   misfit_member which is the misfit for one ensemble member, and
   misfit_ts which is the misfit for one ensemble member / one
   observation key.
*/





#define MISFIT_ENSEMBLE_TYPE_ID        441066
#define MISFIT_MEMBER_TYPE_ID         541066
#define MISFIT_TS_TYPE_ID       641066




struct misfit_ensemble_struct {
  UTIL_TYPE_ID_DECLARATION;
  bool                  initialized;
  int                   history_length;  
  vector_type         * ensemble;           /* Vector of misfit_member_type instances - one for each ensemble member. */
  char                * current_case;       /* The (filesystem) case which was used when building the misfit table. */
};




struct misfit_member_struct {
  UTIL_TYPE_ID_DECLARATION;
  int          my_iens;
  hash_type   *obs;           /* hash table of misfit_ts_type instances - indexed by observation keys. The structure
                                 of this hash table is duplicated for each ensemble member.*/
};



struct misfit_ts_struct {
  UTIL_TYPE_ID_DECLARATION;
  double_vector_type  * data;    /* A double vector of length 'history_length' with actual misfit values. */
};



/******************************************************************/
/* 
   Implementation of the misfit_ts type. Contains the full
   timeseries of misfit for one member/one observation key. The
   implementation is fully static.
*/

static misfit_ts_type * misfit_ts_alloc(int history_length) {
  misfit_ts_type * misfit_ts = util_malloc( sizeof * misfit_ts );
  UTIL_TYPE_ID_INIT(misfit_ts , MISFIT_TS_TYPE_ID);

  if (history_length > 0)
    misfit_ts->data = double_vector_alloc( history_length + 1 , 0 );
  else
    misfit_ts->data = NULL;  /* Used by the xxx_fread_alloc() function below. */

  return misfit_ts;
}


static misfit_ts_type * misfit_ts_buffer_fread_alloc( buffer_type * buffer ) {
  misfit_ts_type * misfit_ts = misfit_ts_alloc( 0 );
  if (misfit_ts->data == NULL)
    misfit_ts->data = double_vector_buffer_fread_alloc( buffer);
  return misfit_ts;
}


static void misfit_ts_buffer_fwrite( const misfit_ts_type * misfit_ts , buffer_type * buffer ) {
  double_vector_buffer_fwrite( misfit_ts->data , buffer );
}


static UTIL_SAFE_CAST_FUNCTION(misfit_ts , MISFIT_TS_TYPE_ID);

static void misfit_ts_free( misfit_ts_type * misfit_ts) {
  double_vector_free( misfit_ts->data );
  free( misfit_ts );
}


static void misfit_ts_free__( void * vector ) {
  misfit_ts_free( misfit_ts_safe_cast( vector ));
}


/**
   Will return the sum over the half-open interval [step1, step2).
*/

static double misfit_ts_sum( const misfit_ts_type * vector , int step1 , int step2 ) {
  double sum = 0;
  const double * data = double_vector_get_const_ptr( vector->data );
  for (int step = step1; step < step2; step++) 
    sum += data[step];
  return sum;
}


static void misfit_ts_iset( misfit_ts_type * vector , int time_index , double value ) {
  double_vector_iset( vector->data , time_index , value );
}

/** Step2 is inclusive - what a fucking mess. */
double misfit_ts_eval( const misfit_ts_type * vector , int step1 , int step2 ) {
  double misfit_sum = 0;
  int step;
  
  for (step = step1; step <= step2; step++)
    misfit_sum += double_vector_iget(vector->data , step );
  
  return misfit_sum;
}

/*****************************************************************/





static UTIL_SAFE_CAST_FUNCTION(misfit_member , MISFIT_MEMBER_TYPE_ID);


static void misfit_member_free( misfit_member_type * node ) {
  hash_free( node->obs );
  free( node );
}


static void misfit_member_free__( void * node ) {
  misfit_member_free( misfit_member_safe_cast( node ));
}

static misfit_member_type * misfit_member_alloc(int iens) {
  misfit_member_type * node = util_malloc( sizeof * node );
  UTIL_TYPE_ID_INIT( node , MISFIT_MEMBER_TYPE_ID);
  node->my_iens    = iens;
  node->obs        = hash_alloc();
  return node;
}


static void misfit_member_install_vector( misfit_member_type * node , const char * key , misfit_ts_type * vector ) {
  hash_insert_hash_owned_ref( node->obs, key , vector , misfit_ts_free__ );
}


static misfit_ts_type * misfit_member_safe_get_vector( misfit_member_type * node , const char * obs_key , int history_length) {
  if (!hash_has_key( node->obs , obs_key ))
    misfit_member_install_vector(node , obs_key , misfit_ts_alloc( history_length ) );
  return hash_get( node->obs , obs_key );
}


misfit_ts_type * misfit_member_get_ts( const misfit_member_type * node , const char * obs_key ) {
  return hash_get( node->obs , obs_key );
}

bool misfit_member_has_ts( const misfit_member_type * node , const char * obs_key ) {
  return hash_has_key( node->obs , obs_key );
}


static void misfit_member_update( misfit_member_type * node , const char * obs_key , int history_length , int iens , const double ** work_chi2) {
  misfit_ts_type * vector = misfit_member_safe_get_vector( node , obs_key , history_length );
  for (int step = 0; step <= history_length; step++) 
    misfit_ts_iset( vector , step , work_chi2[step][iens]);
}


static void misfit_member_buffer_fwrite( const misfit_member_type * node , buffer_type * buffer ) {
  buffer_fwrite_int( buffer , node->my_iens );
  buffer_fwrite_int( buffer , hash_get_size( node->obs ));
  {
    hash_iter_type * obs_iter = hash_iter_alloc( node->obs );
    while ( !hash_iter_is_complete( obs_iter )) {
      const char * key                   = hash_iter_get_next_key( obs_iter );
      misfit_ts_type * misfit_ts = hash_get( node->obs , key );
      buffer_fwrite_string( buffer , key );
      misfit_ts_buffer_fwrite( misfit_ts , buffer);
    }
    hash_iter_free( obs_iter );
  }
}


static misfit_member_type * misfit_member_buffer_fread_alloc( buffer_type * buffer ) {
  int my_iens             = buffer_fread_int( buffer );
  misfit_member_type * node = misfit_member_alloc( my_iens );
  int hash_size = buffer_fread_int( buffer );
  {
    int iobs;
    for (iobs = 0; iobs < hash_size; iobs++) {
      const char         * key           = buffer_fread_string( buffer );
      misfit_ts_type * misfit_ts = misfit_ts_buffer_fread_alloc( buffer );
      misfit_member_install_vector( node , key , misfit_ts );
    }
  }
  return node;
}



/*****************************************************************/


static double ** __2d_malloc(int rows , int columns) {
  double ** d = util_calloc( rows , sizeof * d );
  for (int i =0; i < rows; i++)
    d[i] = util_calloc( columns , sizeof * d[i]);
  return d;
}

static void  __2d_free(double ** d , int rows) {
  for (int i =0; i < rows; i++)
    free(d[i]);
  free(d);
}


void misfit_ensemble_update( misfit_ensemble_type * misfit_ensemble , const ensemble_config_type * ensemble_config , const enkf_obs_type * enkf_obs , enkf_fs_type * fs , int ens_size , int history_length) {
  misfit_ensemble_clear( misfit_ensemble );
  {
    state_enum cmp_state           = FORECAST;
    msg_type * msg                 = msg_alloc("Evaluating misfit for observation: " , false);
    double ** chi2_work            = __2d_malloc( history_length + 1 , ens_size );
    bool_vector_type * iens_valid  = bool_vector_alloc( ens_size , true );
    
    hash_iter_type * obs_iter = enkf_obs_alloc_iter( enkf_obs );
    const char * obs_key      = hash_iter_get_next_key( obs_iter );
    
    misfit_ensemble->history_length = history_length;
    misfit_ensemble_set_ens_size( misfit_ensemble , ens_size );
    
    msg_show( msg );
    while (obs_key != NULL) {
      obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_obs , obs_key );
      msg_update( msg , obs_key );
      
      bool_vector_reset( iens_valid );
      bool_vector_iset( iens_valid , ens_size - 1 , true );
      obs_vector_ensemble_chi2( obs_vector , 
                                fs , 
                                iens_valid , 
                                0 , 
                                misfit_ensemble->history_length, 
                                0 , 
                                ens_size , 
                                cmp_state , 
                                chi2_work);
      
      /** 
          Internalizing the results from the chi2_work table into the misfit structure.
      */
      for (int iens = 0; iens < ens_size; iens++) {
        misfit_member_type * node = misfit_ensemble_iget_member( misfit_ensemble , iens );
        if (bool_vector_iget( iens_valid , iens))
          misfit_member_update( node , obs_key , misfit_ensemble->history_length , iens , (const double **) chi2_work);
      }
      obs_key = hash_iter_get_next_key( obs_iter );
    }
    
    bool_vector_free( iens_valid );
    msg_free(msg , true );
    hash_iter_free( obs_iter );
    
    __2d_free( chi2_work , misfit_ensemble->history_length + 1);
    misfit_ensemble->initialized = true;
  }
}


void misfit_ensemble_buffer_fwrite( const misfit_ensemble_type * misfit_ensemble , buffer_type * buffer ) {
  int ens_size = vector_get_size( misfit_ensemble->ensemble);
  buffer_fwrite_string(buffer , misfit_ensemble->current_case );
  buffer_fwrite_int( buffer , misfit_ensemble->history_length );
  buffer_fwrite_int( buffer , vector_get_size( misfit_ensemble->ensemble ));

  /* Writing the nodes - one for each ensemble member */
  {
    int iens;
    for (iens = 0; iens < ens_size; iens++) 
      misfit_member_buffer_fwrite( vector_iget( misfit_ensemble->ensemble , iens ) , buffer ); 
  }
  
  /* Does not store rankings currently */
}


/**
   Dumps a misfit table to file.
*/

void misfit_ensemble_fwrite( const misfit_ensemble_type * misfit_ensemble , FILE * stream) {
  buffer_type * buffer = buffer_alloc( 1024 );
  misfit_ensemble_buffer_fwrite( misfit_ensemble , buffer );
  buffer_stream_fwrite( buffer , stream );
  buffer_free( buffer );
}


/**
   Observe that the object is NOT in a valid state when leaving this function, 
   must finalize in either misfit_ensemble_alloc() or misfit_ensemble_fread_alloc().
*/

static misfit_ensemble_type * misfit_ensemble_alloc_empty() {
  misfit_ensemble_type * table    = util_malloc( sizeof * table );

  table->initialized     = false;
  table->ensemble        = vector_alloc_new();
  
  return table;
}


/**
   This funcion is a feeble attempt at allowing the ensemble size to
   change runtime. If the new ensemble size is larger than the current
   ensemble size ALL the currently internalized misfit information is
   dropped on the floor; if the the ensemble is shrinked only the the
   last elements of the misfit table are discarded (NOT exactly battle-tested).

*/
void misfit_ensemble_set_ens_size( misfit_ensemble_type * misfit_ensemble , int ens_size) {
  int iens;
  if (ens_size > vector_get_size( misfit_ensemble->ensemble )) {
    /* The new ensemble is larger than what we have currently internalized, 
       we drop everything and add empty misfit_member instances. */
    vector_clear( misfit_ensemble->ensemble );
    for (iens = 0; iens < ens_size; iens++)
      vector_append_owned_ref( misfit_ensemble->ensemble , misfit_member_alloc( iens ) , misfit_member_free__);
    
  } else 
    /* We shrink the vector by removing the last elements. */
    vector_shrink( misfit_ensemble->ensemble , ens_size);
}



misfit_ensemble_type * misfit_ensemble_fread_alloc( const char * filename , const enkf_obs_type * enkf_obs) {
  misfit_ensemble_type * misfit_ensemble = misfit_ensemble_alloc_empty( enkf_obs );
  buffer_type * buffer = buffer_fread_alloc( filename );
  int ens_size;
  
  misfit_ensemble->current_case   = buffer_fread_alloc_string( buffer );
  misfit_ensemble->history_length = buffer_fread_int( buffer );
  ens_size                     = buffer_fread_int( buffer );
  misfit_ensemble_set_ens_size( misfit_ensemble , ens_size );
  {
    for (int iens = 0; iens < ens_size; iens++) {
      misfit_member_type * node = misfit_member_buffer_fread_alloc( buffer );
      vector_iset_owned_ref( misfit_ensemble->ensemble , iens , node , misfit_member_free__);
    }
  }

  buffer_free( buffer );
  return misfit_ensemble;
}



misfit_ensemble_type * misfit_ensemble_alloc( ) {
  misfit_ensemble_type * table = misfit_ensemble_alloc_empty( );
  table->current_case       = util_alloc_string_copy( "????" );
  
  return table;
}



misfit_member_type * misfit_ensemble_iget_member( const misfit_ensemble_type * table , int iens) {
  return vector_iget( table->ensemble , iens);
}




void misfit_ensemble_clear( misfit_ensemble_type * table) {
  vector_clear( table->ensemble );
  table->initialized = false;
}


void misfit_ensemble_free(misfit_ensemble_type * table ) {
  vector_free( table->ensemble );
  free( table->current_case );
  free( table );
}


bool misfit_ensemble_initialized( const misfit_ensemble_type * misfit_ensemble ) {
  return misfit_ensemble->initialized;
}


/*****************************************************************/


int misfit_ensemble_get_ens_size( const misfit_ensemble_type * misfit_ensemble ) {
  return vector_get_size( misfit_ensemble->ensemble );
}
