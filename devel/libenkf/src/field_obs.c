/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'field_obs.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

/**
   See the overview documentation of the observation system in enkf_obs.c
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <stringlist.h>
#include <enkf_util.h>
#include <field_obs.h> 
#include <field_config.h>
#include <obs_data.h>
#include <meas_data.h>
#include <field_config.h>
#include <field.h>
#include <active_list.h>
#include <stringlist.h>

#define FIELD_OBS_TYPE_ID 661098
#define POINT_OBS_TYPE_ID 778196


typedef struct  {
  UTIL_TYPE_ID_DECLARATION;
  int    i;
  int    j;
  int    k;
  int    active_index;
  double value;
  double std;
} point_obs_type; 



struct field_obs_struct {
  UTIL_TYPE_ID_DECLARATION;
  char   * field_name;                    /** The state keyword for the observed field - PRESSURE / SWAT / PORO /...   */
  char   * obs_key;               /** A user provided label for the observation.      */
  int      size;                          /** The number of field cells observed.             */
  int    * index_list;                    /** The list indices which are observed - (active indices). */
  int    * i;                             /** The vector of indices i,j,k are equivalent to those in index_list - i,j,k are only retained for RFT plotting. */
  int    * j;
  int    * k;
  double * obs_value;                     /** The observed values.                            */
  double * obs_std;                       /** The standard deviation of the observations.     */
  
  const field_config_type * field_config; /* The config object of the field we are observing - shared reference. */
};


static UTIL_SAFE_CAST_FUNCTION_CONST(field_obs , FIELD_OBS_TYPE_ID);
static UTIL_SAFE_CAST_FUNCTION(field_obs , FIELD_OBS_TYPE_ID);
UTIL_IS_INSTANCE_FUNCTION(field_obs , FIELD_OBS_TYPE_ID);

static UTIL_SAFE_CAST_FUNCTION_CONST(point_obs , POINT_OBS_TYPE_ID);
static UTIL_SAFE_CAST_FUNCTION(point_obs , POINT_OBS_TYPE_ID);
UTIL_IS_INSTANCE_FUNCTION(point_obs , POINT_OBS_TYPE_ID);


/*****************************************************************/

static point_obs_type * point_obs_alloc( int i , int j , int k , int active_index , double value , double std) {
  point_obs_type * point_obs = util_malloc( sizeof * point_obs , __func__);
  UTIL_TYPE_ID_INIT( point_obs , POINT_OBS_TYPE_ID );
  point_obs->i            = i;
  point_obs->j            = j;
  point_obs->k            = k;       
  point_obs->active_index = active_index;
  point_obs->value        = value;
  point_obs->std          = std;
  
  return point_obs;
}


static void point_obs_free( point_obs_type * point_obs ) {
  free( point_obs );
}


/*****************************************************************/



/**
   The input vectors i,j,k should contain offset zero values.
*/
field_obs_type * field_obs_alloc(
  const char   * obs_key,
  const field_config_type * field_config , 
  const char   * field_name,
  int            size,
  const int    * i,
  const int    * j,
  const int    * k,
  const double * obs_value,
  const double * obs_std)
{
  
  field_obs_type * field_obs = util_malloc(sizeof * field_obs, __func__);
  UTIL_TYPE_ID_INIT( field_obs , FIELD_OBS_TYPE_ID );
  field_obs->size            = size;
  field_obs->field_name      = util_alloc_string_copy(field_name);
  field_obs->obs_key         = util_alloc_string_copy(obs_key);
  field_obs->index_list      = util_malloc( size * sizeof * field_obs->index_list , __func__);
  field_obs->field_config    = field_config;
  {
    int l;
    for (l = 0; l < size; l++) {
      if (field_config_ijk_valid(field_config , i[l] , j[l] , k[l])) {
        int active_index = field_config_active_index(field_config , i[l] , j[l] , k[l]);
        if (active_index >= 0) 
          field_obs->index_list[l] = active_index;
        else
          util_abort("%s: sorry: cell:(%d,%d,%d) is not active - can not observe it. \n",__func__ , i[l]+1 , j[l]+1 , k[l]+1);
      } else
        util_abort("%s: sorry: cell (%d,%d,%d) is outside valid range:  \n",__func__ , i[l]+1 , j[l]+1 , k[l]+1);
    }
  }
  field_obs->i               = util_alloc_copy(i         , size * sizeof * i                    , __func__);
  field_obs->j               = util_alloc_copy(j         , size * sizeof * j                    , __func__);
  field_obs->k               = util_alloc_copy(k         , size * sizeof * k                    , __func__);
  field_obs->obs_value       = util_alloc_copy(obs_value , size * sizeof * field_obs->obs_value , __func__);
  field_obs->obs_std         = util_alloc_copy(obs_std   , size * sizeof * field_obs->obs_value , __func__);
  
  return field_obs;
}



void field_obs_free(
  field_obs_type * field_obs)
{
  free(field_obs->index_list);
  free(field_obs->obs_value);
  free(field_obs->obs_std);
  free(field_obs->field_name);
  free(field_obs->obs_key);
  free(field_obs->i);
  free(field_obs->j);
  free(field_obs->k);
  free(field_obs);
}






const char * field_obs_get_field_name(
  const field_obs_type * field_obs)
{
  return field_obs->field_name;
}




void field_obs_get_observations(const field_obs_type * field_obs,  obs_data_type * obs_data,  int report_step , const active_list_type * __active_list) {
  int i;
  int active_size              = active_list_get_active_size( __active_list , field_obs->size );
  active_mode_type active_mode = active_list_get_mode( __active_list );
  obs_block_type * obs_block   = obs_data_add_block( obs_data , field_obs->obs_key , field_obs->size , NULL , false );
  
  if (active_mode == ALL_ACTIVE) {
    for (i=0; i < field_obs->size; i++) 
      obs_block_iset(obs_block , i , field_obs->obs_value[i] , field_obs->obs_std[i] );
    
  } else if (active_mode == PARTLY_ACTIVE) {
    const int   * active_list    = active_list_get_active( __active_list ); 
    for (i =0 ; i < active_size; i++) {
      int iobs = active_list[i];
      obs_block_iset(obs_block , iobs , field_obs->obs_value[iobs] , field_obs->obs_std[iobs] );
    }
  }
}




void field_obs_measure(const field_obs_type * field_obs, const field_type * field_state, int report_step , int iens , meas_data_type * meas_data , const active_list_type * __active_list) {
  int active_size = active_list_get_active_size( __active_list , field_obs->size );
  meas_block_type * meas_block = meas_data_add_block( meas_data , field_obs->obs_key , report_step , field_obs->size );
  int iobs;

  active_mode_type active_mode = active_list_get_mode( __active_list );
  if (active_mode == ALL_ACTIVE) {
    for (iobs=0; iobs < field_obs->size; iobs++) {
      double value = field_iget_double(field_state , field_obs->index_list[iobs]);
      meas_block_iset( meas_block , iens , iobs , value );
    }
  } else if (active_mode == PARTLY_ACTIVE) {
    const int   * active_list    = active_list_get_active( __active_list ); 
    for (int i =0 ; i < active_size; i++) {
      iobs = active_list[i];
      double value = field_iget_double(field_state , field_obs->index_list[iobs]);
      meas_block_iset( meas_block , iens , field_obs->index_list[ iobs ] , value );
    }
  }
}





double field_obs_chi2(const field_obs_type * field_obs,  const field_type     * field_state) {
  double sum_chi2 = 0;
  for (int i=0; i < field_obs->size; i++) {
    double x = (field_iget_double(field_state , field_obs->index_list[i]) - field_obs->obs_value[i]) / field_obs->obs_std[i];
    sum_chi2 += x*x;
  }
  return sum_chi2;
}




/**
   The index is into the the number of active cells which are observed by this observation.
*/
void field_obs_iget(const field_obs_type * field_obs, int index , double *value , double * std) {
  *value = field_obs->obs_value[index];
  *std   = field_obs->obs_std[index];
}


void field_obs_user_get(const field_obs_type * field_obs , const char * index_key , double *value , double * std, bool * valid) {
  int      i,j,k;
  int      parse_user_key = field_config_parse_user_key(field_obs->field_config , index_key , &i, &j , &k);

  *valid = false;
  if (parse_user_key == 0) {
    int active_index = field_config_active_index(field_obs->field_config , i,j,k);
    int l = 0;
    /* iterating through all the cells the observation is observing. */
    while (!(*valid) && l < field_obs->size) {
      if (field_obs->index_list[l] == active_index) {
        *value = field_obs->obs_value[l];
        *std   = field_obs->obs_std[l];
        *valid = true;
      }
      l++;
    }
  }
}




const int * field_obs_get_i(const field_obs_type * field_obs) {
  return field_obs->i;
}

const int * field_obs_get_j(const field_obs_type * field_obs) {
  return field_obs->j;
}

const int * field_obs_get_k(const field_obs_type * field_obs) {
  return field_obs->k;
}


/*
  Returns by reference i,j,k for observation point nr block_nr.
*/

void field_obs_iget_ijk(const field_obs_type * field_obs , int block_nr , int * i , int * j , int * k) {
  *i = field_obs->i[block_nr];
  *j = field_obs->j[block_nr];
  *k = field_obs->k[block_nr];
}


int field_obs_get_size(const field_obs_type * field_obs) {
  return field_obs->size;
}


/*****************************************************************/

VOID_FREE(field_obs)
VOID_GET_OBS(field_obs)
VOID_MEASURE(field_obs , field)
VOID_USER_GET_OBS(field_obs)
VOID_CHI2(field_obs , field)
