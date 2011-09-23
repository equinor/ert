/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
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
#include <string.h>
#include <util.h>
#include <matrix.h>
#include <matrix_blas.h>
#include <stdio.h>
#include <analysis_table.h>
#include <enkf_linalg.h>
#include <std_enkf.h>


/*
  The sqrt_enkf module performs a EnKF update based on the square root
  scheme. Observe that this module shares quite a lot of
  implementation with the std_enkf module.
*/

#define SQRT_ENKF_TYPE_ID 268823

typedef struct {
  UTIL_TYPE_ID_DECLARATION;
  std_enkf_data_type * std_data;
  long                 options;
} sqrt_enkf_data_type;


static UTIL_SAFE_CAST_FUNCTION( sqrt_enkf_data , SQRT_ENKF_TYPE_ID )


void * sqrt_enkf_data_alloc( ) {
  sqrt_enkf_data_type * data = util_malloc( sizeof * data , __func__ );
  UTIL_TYPE_ID_INIT( data , SQRT_ENKF_TYPE_ID );

  data->options  = ANALYSIS_NEED_RANDROT;
  data->std_data = std_enkf_data_alloc( );
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
                     matrix_type * S , 
                     matrix_type * R , 
                     matrix_type * innov , 
                     matrix_type * E , 
                     matrix_type *D, 
                     matrix_type * randrot) {

  sqrt_enkf_data_type * data = sqrt_enkf_data_safe_cast( module_data );
  {
    int ncomp         = std_enkf_get_subspace_dimension( data->std_data );
    double truncation = std_enkf_get_truncation( data->std_data );
    int nrobs         = matrix_get_rows( S );
    int ens_size      = matrix_get_columns( S );
    int nrmin         = util_int_min( ens_size , nrobs); 
    matrix_type * W   = matrix_alloc(nrobs , nrmin);                      
    double      * eig = util_malloc( sizeof * eig * nrmin , __func__);    
    
    enkf_linalg_lowrankCinv( S , R , W , eig , truncation , ncomp);    
    { /* The part in the block here was a seperate function, and might be
         factored out again. */
      matrix_type * X2    = matrix_alloc(nrmin , ens_size);
      
      //if (bootstrap)
      //util_exit("%s: Sorry bootstrap support not fully implemented for SQRT scheme\n",__func__);
      
      enkf_linalg_meanX5( S , W , eig , innov , X );
      enkf_linalg_genX2(X2 , S , W , eig);
      enkf_linalg_X5sqrt(X2 , X , randrot , nrobs);
      
      matrix_free( X2 );
    }
    matrix_free( W );
    free( eig );
  }
}


bool sqrt_enkf_get_option( void * arg , long flag ) {
  sqrt_enkf_data_type * module_data = sqrt_enkf_data_safe_cast( arg );
  {
    return (flag & module_data->options);
  }
}





/*****************************************************************/


#ifdef INTERNAL_LINK
#define SYMBOL_TABLE sqrt_enkf_symbol_table
#else
#define SYMBOL_TABLE analysis_table
#endif

analysis_table_type SYMBOL_TABLE[] = {
  { 
    .alloc           = sqrt_enkf_data_alloc,
    .freef           = sqrt_enkf_data_free,
    .set_int         = sqrt_enkf_set_int , 
    .set_double      = sqrt_enkf_set_double , 
    .set_string      = NULL , 
    .initX           = sqrt_enkf_initX , 
    .init_update     = NULL,
    .complete_update = NULL,
    .get_option      = sqrt_enkf_get_option , 
  }
};
