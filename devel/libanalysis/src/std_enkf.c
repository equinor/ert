/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'std_enkf.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#define STD_ENKF_TYPE_ID 261123

#define INVALID_SUBSPACE_DIMENSION  -1
#define INVALID_TRUNCATION          -1
#define DEFAULT_SUBSPACE_DIMENSION  INVALID_SUBSPACE_DIMENSION

/*
  Observe that only one of the settings subspace_dimension and
  truncation can be valid at a time; otherwise the svd routine will
  fail. This implies that the set_truncation() and
  set_subspace_dimension() routines will set one variable, AND
  INVALIDATE THE OTHER. For most situations this will be OK, but if
  you have repeated calls to both of these functions the end result
  might be a surprise.  
*/




struct std_enkf_data_struct {
  UTIL_TYPE_ID_DECLARATION;
  double    truncation;            // ENKF_TRUNCATION_KEY
  int       subspace_dimension;    // ENKF_NCOMP_KEY (-1: use Truncation instead)
};



static UTIL_SAFE_CAST_FUNCTION( std_enkf_data , STD_ENKF_TYPE_ID )


void std_enkf_set_truncation( std_enkf_data_type * data , double truncation ) {
  data->truncation = truncation;
  if (truncation > 0.0)
    data->subspace_dimension = INVALID_SUBSPACE_DIMENSION;
}

double std_enkf_get_truncation( void * module_data ) {
  std_enkf_data_type * data = std_enkf_data_safe_cast( module_data );
  return data->truncation;
}

void std_enkf_set_subspace_dimension( std_enkf_data_type * data , int subspace_dimension) {
  data->subspace_dimension = subspace_dimension;
  if (subspace_dimension > 0)
    data->truncation = INVALID_TRUNCATION;
}

int std_enkf_get_subspace_dimension( void * module_data ) {
  std_enkf_data_type * data = std_enkf_data_safe_cast( module_data );
  return data->subspace_dimension;
}


void * std_enkf_data_alloc( ) {
  std_enkf_data_type * data = util_malloc( sizeof * data , __func__ );
  UTIL_TYPE_ID_INIT( data , STD_ENKF_TYPE_ID );

  std_enkf_set_truncation( data , DEFAULT_ENKF_TRUNCATION_ );
  std_enkf_set_subspace_dimension( data , DEFAULT_SUBSPACE_DIMENSION );
  
  return data;
}


void std_enkf_data_free( void * data ) { 
  free( data );
}


void std_enkf_initX(void * module_data , 
                    matrix_type * X , 
                    matrix_type * S , 
                    matrix_type * R , 
                    matrix_type * innov , 
                    matrix_type * E , 
                    matrix_type *D, 
                    matrix_type * randrot) {

  std_enkf_data_type * data = std_enkf_data_safe_cast( module_data );
  {
    int ncomp         = data->subspace_dimension;
    double truncation = data->truncation;
    int nrobs         = matrix_get_rows( S );
    int ens_size      = matrix_get_columns( S );
    int nrmin         = util_int_min( ens_size , nrobs); 
    matrix_type * W   = matrix_alloc(nrobs , nrmin);                      
    double      * eig = util_malloc( sizeof * eig * nrmin , __func__);    
    
    enkf_linalg_lowrankCinv( S , R , W , eig , truncation , ncomp);    
    { /* The part in the block here was a seperate function, and might be
         factored out again. */
      matrix_type * X3  = matrix_alloc(nrobs , ens_size);
      enkf_linalg_genX3(X3 , W , D , eig ); /*  X2 = diag(eig) * W' * D (Eq. 14.31, Evensen (2007)) */
                                            /*  X3 = W * X2 = X1 * X2 (Eq. 14.31, Evensen (2007)) */  

      matrix_dgemm( X , S , X3 , true , false , 1.0 , 0.0);  /* X = S' * X3 */
      // If !bootstrap {
      for (int i = 0; i < ens_size ; i++)
        matrix_iadd( X , i , i , 1.0); /*X = I + X */
      // }

      matrix_free( X3 );

    }
    matrix_free( W );
    free( eig );
  }
}



bool std_enkf_set_double( void * arg , const char * var_name , double value) {
  std_enkf_data_type * module_data = std_enkf_data_safe_cast( arg );
  {
    bool name_recognized = true;

    if (strcmp( var_name , ENKF_TRUNCATION_KEY_) == 0)
      std_enkf_set_truncation( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}


bool std_enkf_set_int( void * arg , const char * var_name , int value) {
  std_enkf_data_type * module_data = std_enkf_data_safe_cast( arg );
  {
    bool name_recognized = true;
    
    if (strcmp( var_name , ENKF_NCOMP_KEY_) == 0)
      std_enkf_set_subspace_dimension( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}




/**
   gcc -fpic -c <object_file> -I??  <src_file>
   gcc -shared -o <lib_file> <object_files>
*/



#ifdef INTERNAL_LINK
#define SYMBOL_TABLE std_enkf_symbol_table
#else
#define SYMBOL_TABLE analysis_table
#endif

analysis_table_type SYMBOL_TABLE[] = {
  { 
    .alloc        = std_enkf_data_alloc,
    .freef        = std_enkf_data_free,
    .set_int      = std_enkf_set_int , 
    .set_double   = std_enkf_set_double , 
    .set_string   = NULL , 
    .initX        = std_enkf_initX , 
    .need_ED      = true,
    .need_randrot = false,
  }
};

