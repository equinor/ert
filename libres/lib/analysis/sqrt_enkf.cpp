/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'sqrt_enkf.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/util.hpp>
#include <ert/res_util/matrix.hpp>

#include <ert/analysis/analysis_table.hpp>
#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/enkf_linalg.hpp>
#include <ert/analysis/std_enkf.hpp>


/*
  The sqrt_enkf module performs a EnKF update based on the square root
  scheme. Observe that this module shares quite a lot of
  implementation with the std_enkf module.
*/

#define SQRT_ENKF_TYPE_ID 268823

typedef struct {
  UTIL_TYPE_ID_DECLARATION;
  std_enkf_data_type * std_data;
  matrix_type        * randrot;
  long                 options;
} sqrt_enkf_data_type;


static UTIL_SAFE_CAST_FUNCTION( sqrt_enkf_data , SQRT_ENKF_TYPE_ID )
static UTIL_SAFE_CAST_FUNCTION_CONST( sqrt_enkf_data , SQRT_ENKF_TYPE_ID )


void * sqrt_enkf_data_alloc( ) {
  sqrt_enkf_data_type * data = (sqrt_enkf_data_type*)util_malloc( sizeof * data );
  UTIL_TYPE_ID_INIT( data , SQRT_ENKF_TYPE_ID );

  data->std_data = (std_enkf_data_type*)std_enkf_data_alloc( );
  data->randrot  = NULL;
  data->options  = ANALYSIS_SCALE_DATA;

  return data;
}



void sqrt_enkf_data_free( void * data ) {
  sqrt_enkf_data_type * module_data = sqrt_enkf_data_safe_cast( data );
  {
    std_enkf_data_free( module_data->std_data );
    free( module_data );
  }
}



bool sqrt_enkf_set_double( void * arg , const char * var_name , double value) {
  sqrt_enkf_data_type * module_data = sqrt_enkf_data_safe_cast( arg );
  {
    if (std_enkf_set_double( module_data->std_data , var_name , value ))
      return true;
    else {
      /* Could in principle set sqrt specific variables here. */
      return false;
    }
  }
}


bool sqrt_enkf_set_int( void * arg , const char * var_name , int value) {
  sqrt_enkf_data_type * module_data = sqrt_enkf_data_safe_cast( arg );
  {
    if (std_enkf_set_int( module_data->std_data , var_name , value ))
      return true;
    else {
      /* Could in principle set sqrt specific variables here. */
      return false;
    }
  }
}





void sqrt_enkf_initX(void * module_data ,
                     matrix_type * X ,
                     const matrix_type * A ,
                     const matrix_type * S0 ,
                     const matrix_type * R ,
                     const matrix_type * dObs ,
                     const matrix_type * E ,
                     const matrix_type *D,
                     rng_type * rng) {

  sqrt_enkf_data_type * data = sqrt_enkf_data_safe_cast( module_data );
  {
    matrix_type * S   = matrix_alloc_copy(S0);
    int ncomp         = std_enkf_get_subspace_dimension( data->std_data );
    double truncation = std_enkf_get_truncation( data->std_data );
    int nrobs         = matrix_get_rows( S );
    int ens_size      = matrix_get_columns( S );
    int nrmin         = util_int_min( ens_size , nrobs);
    matrix_type * W   = matrix_alloc(nrobs , nrmin);
    double      * eig = (double*)util_calloc( nrmin , sizeof * eig );

    matrix_subtract_row_mean( S );   /* Shift away the mean */
    enkf_linalg_lowrankCinv( S , R , W , eig , truncation , ncomp);
    enkf_linalg_init_sqrtX( X , S , data->randrot , dObs , W , eig , false);
    matrix_free( W );
    free( eig );

    enkf_linalg_checkX( X , false );
    matrix_free(S);
  }
}


long sqrt_enkf_get_options( void * arg , long flag ) {
  sqrt_enkf_data_type * module_data = sqrt_enkf_data_safe_cast( arg );
  {
    return module_data->options;
  }
}



// Called from analysis_module.c: analysis_module_init_update()
void sqrt_enkf_init_update( void * arg ,
                          const bool_vector_type * ens_mask,
                          const bool_vector_type * obs_mask,
                          const matrix_type * S ,
                          const matrix_type * R ,
                          const matrix_type * dObs ,
                          const matrix_type * E ,
                          const matrix_type * D,
                          rng_type * rng ) {

  sqrt_enkf_data_type * sqrt_data = sqrt_enkf_data_safe_cast( arg );
  {
    int ens_size = matrix_get_columns( S );
    sqrt_data->randrot = enkf_linalg_alloc_mp_randrot( ens_size , rng );
  }
}


void sqrt_enkf_complete_update( void * arg ) {
  sqrt_enkf_data_type * sqrt_data = sqrt_enkf_data_safe_cast( arg );
  {
    matrix_free( sqrt_data->randrot );
    sqrt_data->randrot = NULL;
  }
}

bool sqrt_enkf_has_var( const void * arg, const char * var_name) {
    const sqrt_enkf_data_type * module_data = sqrt_enkf_data_safe_cast_const( arg );
    {
      return std_enkf_has_var(module_data->std_data, var_name);
    }
}

double sqrt_enkf_get_double( const void * arg, const char * var_name) {
    const sqrt_enkf_data_type * module_data = sqrt_enkf_data_safe_cast_const( arg );
    {
      return std_enkf_get_double( module_data->std_data , var_name);
    }
}

int sqrt_enkf_get_int( const void * arg, const char * var_name) {
    const sqrt_enkf_data_type * module_data = sqrt_enkf_data_safe_cast_const( arg );
    {
      return std_enkf_get_int( module_data->std_data , var_name);
    }
}



/*****************************************************************/

#ifdef INTERNAL_LINK
#define LINK_NAME SQRT_ENKF
#else
#define LINK_NAME EXTERNAL_MODULE_SYMBOL
#endif


analysis_table_type LINK_NAME = {
  .name            = "SQRT_ENKF",
  .updateA         = NULL,
  .initX           = sqrt_enkf_initX ,
  .init_update     = sqrt_enkf_init_update,
  .complete_update = sqrt_enkf_complete_update,

  .freef           = sqrt_enkf_data_free,
  .alloc           = sqrt_enkf_data_alloc,

  .set_int         = sqrt_enkf_set_int ,
  .set_double      = sqrt_enkf_set_double ,
  .set_bool        = NULL ,
  .set_string      = NULL ,
  .get_options     = sqrt_enkf_get_options,

  .has_var         = sqrt_enkf_has_var,
  .get_int         = sqrt_enkf_get_int,
  .get_double      = sqrt_enkf_get_double,
  .get_bool        = NULL,
  .get_ptr         = NULL
};
