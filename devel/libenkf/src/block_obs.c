/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'block_obs.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ecl_sum.h>
#include <ecl_grid.h>
#include <smspec_node.h>

#include <enkf_util.h>
#include <field_config.h>
#include <obs_data.h>
#include <meas_data.h>
#include <field_config.h>
#include <field.h>
#include <active_list.h>
#include <block_obs.h> 
#include <enkf_defaults.h>

#define BLOCK_OBS_TYPE_ID 661098
#define POINT_OBS_TYPE_ID 778196


typedef struct  {
  UTIL_TYPE_ID_DECLARATION;
  int      i;
  int      j;
  int      k;
  int      active_index;
  double   value;
  double   std;
  char   * sum_key;
} point_obs_type; 



struct block_obs_struct {
  UTIL_TYPE_ID_DECLARATION;
  char   * field_name;                    /** The state keyword for the observed field - PRESSURE / SWAT / PORO /...   */
  char   * obs_key;                       /** A user provided label for the observation.      */
  int      size;                          /** The number of field cells observed.             */
  point_obs_type ** point_list;
  
  const ecl_grid_type     * grid;
  const field_config_type * field_config; /* The config object of the field we are observing - shared reference. */
  block_obs_source_type source_type;
};


static UTIL_SAFE_CAST_FUNCTION_CONST(block_obs , BLOCK_OBS_TYPE_ID);
static UTIL_SAFE_CAST_FUNCTION(block_obs , BLOCK_OBS_TYPE_ID);
UTIL_IS_INSTANCE_FUNCTION(block_obs , BLOCK_OBS_TYPE_ID);

/*
static UTIL_SAFE_CAST_FUNCTION_CONST(point_obs , POINT_OBS_TYPE_ID);
static UTIL_SAFE_CAST_FUNCTION(point_obs , POINT_OBS_TYPE_ID);
UTIL_IS_INSTANCE_FUNCTION(point_obs , POINT_OBS_TYPE_ID);
*/

/*****************************************************************/






static point_obs_type * point_obs_alloc( int i , int j , int k , int active_index , char * sum_key , double value , double std) {
  point_obs_type * point_obs = util_malloc( sizeof * point_obs , __func__);
  UTIL_TYPE_ID_INIT( point_obs , POINT_OBS_TYPE_ID );
  point_obs->i            = i;
  point_obs->j            = j;
  point_obs->k            = k;       
  point_obs->active_index = active_index;
  point_obs->value        = value;
  point_obs->std          = std;
  point_obs->sum_key      = sum_key;
  return point_obs;
}




static void point_obs_free( point_obs_type * point_obs ) {
  util_safe_free( point_obs->sum_key );
  free( point_obs );
}


static double point_obs_measure( const point_obs_type * point_obs , const field_type * field_state) {
  return field_iget_double(field_state , point_obs->active_index);
}



/*****************************************************************/


static const char * __summary_kw( const char * field_name ) {
  if (strcmp( field_name , "PRESSURE") == 0)
    return "BPR";
  else if (strcmp( field_name , "SWAT") == 0)
    return "BSWAT";
  else if (strcmp( field_name , "SGAS") == 0)
    return "BSGAS";
  else
    util_abort("%s: sorry - could not \'translate\' field:%s to block summayr variable\n",__func__ , field_name);
}

static void block_obs_validate_ijk( const ecl_grid_type * grid , int size, const int * i , const int * j , const int * k) {
  int l;
  for (l = 0; l < size; l++) {
    if (ecl_grid_ijk_valid(grid , i[l] , j[l] , k[l])) {
      int active_index = ecl_grid_get_active_index3( grid , i[l] , j[l] , k[l]);
      if (active_index < 0) 
        util_abort("%s: sorry: cell:(%d,%d,%d) is not active - can not observe it. \n",__func__ , i[l]+1 , j[l]+1 , k[l]+1);
      
    } else
      util_abort("%s: sorry: cell (%d,%d,%d) is outside valid range:  \n",__func__ , i[l]+1 , j[l]+1 , k[l]+1);
  }
}

static void block_obs_resize( block_obs_type * block_obs , int new_size) {
  int i;
  block_obs->point_list = util_realloc( block_obs->point_list , new_size * sizeof * block_obs->point_list , __func__);

  for (i=block_obs->size; i < new_size; i++)
    block_obs->point_list[i] = NULL;

  block_obs->size = new_size;
}

/**
   The input vectors i,j,k should contain offset zero values.
*/
block_obs_type * block_obs_alloc(const char   * obs_key,
                                 block_obs_source_type source_type , 
                                 const ecl_grid_type * grid , 
                                 const ecl_sum_type  * refcase , 
                                 const field_config_type * field_config , 
                                 const char   * field_name,
                                 int            size,
                                 const int    * i,
                                 const int    * j,
                                 const int    * k,
                                 const double * obs_value,
                                 const double * obs_std)
{
  block_obs_validate_ijk( grid , size , i,j,k);
  
  {
    block_obs_type * block_obs = util_malloc(sizeof * block_obs, __func__);
    char           * sum_kw    = NULL;
    bool             OK        = true; 

    UTIL_TYPE_ID_INIT( block_obs , BLOCK_OBS_TYPE_ID );
    block_obs->field_name      = util_alloc_string_copy(field_name);
    block_obs->obs_key         = util_alloc_string_copy(obs_key);
    block_obs->field_config    = field_config;
    block_obs->source_type     = source_type; 
    block_obs->size            = 0;
    block_obs->point_list      = NULL;
    block_obs->grid            = grid;
    block_obs_resize( block_obs , size );

    {
      char * join_string = NULL;
      if (source_type == SOURCE_SUMMARY) 
        sum_kw = __summary_kw( field_name );

      for (int l=0; l < size; l++) {
        int active_index = ecl_grid_get_active_index3( grid , i[l],j[l],k[l]);
        char * sum_key   = NULL;

        if (source_type == SOURCE_SUMMARY) {
          sum_key = smspec_alloc_block_ijk_key( SUMMARY_KEY_JOIN_STRING , sum_kw , i[l] + 1 , j[l] + 1 , k[l] + 1);
          
          // Must call: ensemble_config_add_summary(ens_config , sum_key) to
          // ensure that the new sum_key instances are added to the ensemble.
          
          
          // The point_obs instance takes ownership of the sum_key pointer and discards it
          // when going out of scope.
          if (refcase != NULL) {
            if (!ecl_sum_has_key(refcase , sum_key)) {
              fprintf(stderr,"** Warning missing summary %s for cell: (%d,%d,%d) in refcase - observation:%s not added\n" , 
                      sum_kw , i[l]+1 , j[l]+1 , k[l]+1 , block_obs->obs_key );
              free( sum_key );
              sum_key = NULL;

              OK = false;
            }
          }
        } 
        block_obs->point_list[l] = point_obs_alloc(i[l] , j[l] , k[l] , active_index , sum_key , obs_value[l] , obs_std[l]);
      }
    }

    if (!OK) {
      block_obs_free( block_obs );
      block_obs = NULL;
    }
    return block_obs;
  }
}



void block_obs_free( block_obs_type * block_obs) {
  for (int i=0; i < block_obs->size; i++) {
    if (block_obs->point_list[i] != NULL)
      point_obs_free( block_obs->point_list[i]);
  }
  
  util_safe_free(block_obs->point_list );
  free(block_obs->field_name);
  free(block_obs->obs_key);
  free(block_obs);
}







void block_obs_get_observations(const block_obs_type * block_obs,  obs_data_type * obs_data,  int report_step , const active_list_type * __active_list) {
  int i;
  int active_size              = active_list_get_active_size( __active_list , block_obs->size );
  active_mode_type active_mode = active_list_get_mode( __active_list );
  obs_block_type * obs_block   = obs_data_add_block( obs_data , block_obs->obs_key , block_obs->size , NULL , false );
  
  if (active_mode == ALL_ACTIVE) {
    for (i=0; i < block_obs->size; i++) {
      const point_obs_type * point_obs = block_obs->point_list[i];
      obs_block_iset(obs_block , i , point_obs->value , point_obs->std );
    }
  } else if (active_mode == PARTLY_ACTIVE) {
    const int   * active_list    = active_list_get_active( __active_list ); 
    for (i =0 ; i < active_size; i++) {
      int iobs = active_list[i];
      const point_obs_type * point_obs = block_obs->point_list[iobs];
      obs_block_iset(obs_block , iobs , point_obs->value , point_obs->std );
    }
  }
}




static void block_obs_measure_field(const block_obs_type * block_obs, const field_type * field_state , node_id_type node_id , meas_data_type * meas_data , const active_list_type * __active_list) {
  int active_size = active_list_get_active_size( __active_list , block_obs->size );
  meas_block_type * meas_block = meas_data_add_block( meas_data , block_obs->obs_key , node_id.report_step , block_obs->size );
  int iobs;

  active_mode_type active_mode = active_list_get_mode( __active_list );
  if (active_mode == ALL_ACTIVE) {
    for (iobs=0; iobs < block_obs->size; iobs++) {
      const point_obs_type * point_obs = block_obs->point_list[iobs];
      double value = point_obs_measure( point_obs , field_state );
      meas_block_iset( meas_block , node_id.iens , iobs , value );
    }
  } else if (active_mode == PARTLY_ACTIVE) {
    const int   * active_list    = active_list_get_active( __active_list ); 
    for (int i =0 ; i < active_size; i++) {
      iobs = active_list[i];
      {
        const point_obs_type * point_obs = block_obs->point_list[iobs];
        double value = point_obs_measure( point_obs , field_state );
        meas_block_iset( meas_block , node_id.iens , point_obs->active_index , value );
      }
    }
  }
}


static void block_obs_measure_summary(const block_obs_type * block_obs, const field_type * field_state , node_id_type node_id , meas_data_type * meas_data , const active_list_type * __active_list) {
  util_exit("%s: not implemented \n",__func__);
}


void block_obs_measure(const block_obs_type * block_obs, const void * state , node_id_type node_id , meas_data_type * meas_data , const active_list_type * __active_list) {
  if (field_is_instance( state ))
    block_obs_measure_field( block_obs , state , node_id , meas_data , __active_list);
  else
    block_obs_measure_summary( block_obs , state , node_id , meas_data , __active_list);
}


double block_obs_chi2(const block_obs_type * block_obs,  const field_type * field_state, node_id_type node_id) {
  double sum_chi2 = 0;
  for (int i=0; i < block_obs->size; i++) {
    const point_obs_type * point_obs = block_obs->point_list[i];
    double sim_value = point_obs_measure( point_obs , field_state );
    double x = (sim_value - point_obs->value) / point_obs->std;
    sum_chi2 += x*x;
  }
  return sum_chi2;
}




/**
   The index is into the the number of active cells which are observed by this observation.
*/
void block_obs_iget(const block_obs_type * block_obs, int index , double *value , double * std) {
  const point_obs_type * point_obs = block_obs->point_list[index];
  *value = point_obs->value;
  *std   = point_obs->std;
}


void block_obs_user_get(const block_obs_type * block_obs , const char * index_key , double *value , double * std, bool * valid) {
  int      i,j,k;

  *valid = false;
  if (field_config_parse_user_key__( index_key , &i , &j , &k)) {
    int active_index = ecl_grid_get_active_index3(block_obs->grid , i,j,k);
    int l = 0;
    /* iterating through all the cells the observation is observing. */

    while (!(*valid) && l < block_obs->size) {
      const point_obs_type * point_obs = block_obs->point_list[l];
      if (point_obs->active_index == active_index) {
        *value = point_obs->value;
        *std   = point_obs->std;
        *valid = true;
      }
      l++;
    }

  }
}




int block_obs_iget_i(const block_obs_type * block_obs, int index) {
  const point_obs_type * point_obs = block_obs->point_list[index];
  return point_obs->i;
}

int block_obs_iget_j(const block_obs_type * block_obs, int index) {
  const point_obs_type * point_obs = block_obs->point_list[index];
  return point_obs->j;
}

int block_obs_iget_k(const block_obs_type * block_obs, int index) {
  const point_obs_type * point_obs = block_obs->point_list[index];
  return point_obs->k;
}


/*
  Returns by reference i,j,k for observation point nr block_nr.
*/

void block_obs_iget_ijk(const block_obs_type * block_obs , int block_nr , int * i , int * j , int * k) {
  const point_obs_type * point_obs = block_obs->point_list[block_nr];
  *i = point_obs->i;
  *j = point_obs->j;
  *k = point_obs->k;
}


int block_obs_get_size(const block_obs_type * block_obs) {
  return block_obs->size;
}


/*****************************************************************/

VOID_FREE(block_obs)
VOID_GET_OBS(block_obs)
VOID_MEASURE(block_obs , field)
VOID_USER_GET_OBS(block_obs)
VOID_CHI2(block_obs , field)
