/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'cv_enkf.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#define CV_ENKF_TYPE_ID 765523

#define INVALID_SUBSPACE_DIMENSION  -1
#define INVALID_TRUNCATION          -1
#define DEFAULT_SUBSPACE_DIMENSION  INVALID_SUBSPACE_DIMENSION


typedef struct {
  UTIL_TYPE_ID_DECLARATION;
  matrix_type          * Z;
  matrix_type          * Rp;
  matrix_type          * Dp;
  double                 truncation;
  int                    subspace_dimension;  // ENKF_NCOMP_KEY (-1: use Truncation instead)
  long                   option_flags;
} cv_enkf_data_type;



static UTIL_SAFE_CAST_FUNCTION( cv_enkf_data , CV_ENKF_TYPE_ID )

void cv_enkf_set_truncation( cv_enkf_data_type * data , double truncation ) {
  data->truncation = truncation;
  if (truncation > 0.0)
    data->subspace_dimension = INVALID_SUBSPACE_DIMENSION;
}


void cv_enkf_set_subspace_dimension( cv_enkf_data_type * data , int subspace_dimension) {
  data->subspace_dimension = subspace_dimension;
  if (subspace_dimension > 0)
    data->truncation = INVALID_TRUNCATION;
}

void * cv_enkf_data_alloc( ) {
  cv_enkf_data_type * data = util_malloc( sizeof * data , __func__ );
  UTIL_TYPE_ID_INIT( data , CV_ENKF_TYPE_ID );

  data->Z            = NULL;
  data->Rp           = NULL;
  data->Dp           = NULL;
  data->option_flags = ANALYSIS_NEED_ED;
  cv_enkf_set_truncation( data , DEFAULT_ENKF_TRUNCATION_ );
  
  return data;
}



void cv_enkf_data_free( void * arg ) {
  cv_enkf_data_type * cv_data = cv_enkf_data_safe_cast( arg );
  {
    matrix_safe_free( cv_data->Z );
    matrix_safe_free( cv_data->Rp );
    matrix_safe_free( cv_data->Dp );
  }
}






void cv_enkf_init_update( void * arg , 
                          const matrix_type * S , 
                          const matrix_type * R , 
                          const matrix_type * innov , 
                          const matrix_type * E , 
                          const matrix_type * D ) {

  cv_enkf_data_type * cv_data = cv_enkf_data_safe_cast( arg );
  {
    int i, j;
    const int nrobs = matrix_get_rows( S );
    const int nrens = matrix_get_columns( S );
    const int nrmin = util_int_min( nrobs , nrens );

    cv_data->Z  = matrix_alloc( nrmin , nrens );
    cv_data->Rp = matrix_alloc( nrmin , nrmin );
    cv_data->Dp = matrix_alloc( nrmin , nrens );
    
    /*
      Compute SVD(S)
    */
    matrix_type * U0   = matrix_alloc( nrobs , nrmin    ); /* Left singular vectors.  */
    matrix_type * V0T  = matrix_alloc( nrmin , nrens );    /* Right singular vectors. */
    
    double * inv_sig0  = util_malloc( nrmin * sizeof * inv_sig0 , __func__);
    double * sig0      = inv_sig0;

    enkf_linalg_svdS(S , cv_data->truncation , cv_data->subspace_dimension , DGESVD_MIN_RETURN , inv_sig0 , U0 , V0T);
    
    /* Need to use the original non-inverted singular values. */
    for(i = 0; i < nrmin; i++) 
      if ( inv_sig0[i] > 0 ) 
        sig0[i] = 1.0 / inv_sig0[i];
    
    /*
      Compute the actual principal components, Z = sig0 * VOT 
      NOTE: Z contains potentially alot of redundant zeros, but 
      we do not care about this for now
    */
    
    for(i = 0; i < nrmin; i++) 
      for(j = 0; j < nrens; j++) 
        matrix_iset( cv_data->Z , i , j , sig0[i] * matrix_iget( V0T , i , j ) );
    
    /* Also compute Rp */
    {
      matrix_type * X0 = matrix_alloc( nrmin , matrix_get_rows( R ));
      matrix_dgemm(X0 , U0 , R  , true  , false , 1.0 , 0.0);   /* X0 = U0^T * R */
      matrix_dgemm(cv_data->Rp  , X0 , U0 , false , false , 1.0 , 0.0);  /* Rp = X0 * U0 */
      matrix_free(X0);
    }

    /*We also need to compute the reduced "Innovation matrix" Dp = U0' * D    */
    matrix_dgemm(cv_data->Dp , U0 , D , true , false , 1.0 , 0.0);
    
    
    free(inv_sig0);
    matrix_free(U0);
    matrix_free(V0T);
    
    /* 
       2: Diagonalize the S matrix; singular vectors etc. needed later in the local CV:
       (V0T = transposed right singular vectors of S, Z = scaled principal components, 
       eig = scaled, inverted singular vectors, U0 = left singular vectors of S
       eig = inv(I+Lambda1),(Eq.14.30, and 14.29, Evensen, 2007, respectively)
    */ 
  }
}

void cv_enkf_initX(void * module_data , 
                   matrix_type * X , 
                   matrix_type * S , 
                   matrix_type * R , 
                   matrix_type * innov , 
                   matrix_type * E , 
                   matrix_type *D, 
                   matrix_type * randrot) {
  
  cv_enkf_data_type * cv_data = cv_enkf_data_safe_cast( module_data );
  {
    matrix_diag_set_scalar( X , 1.0 );
  }
}



void cv_enkf_complete_update( void * arg ) {
  cv_enkf_data_type * cv_data = cv_enkf_data_safe_cast( arg );
  {
    matrix_free( cv_data->Z  );
    matrix_free( cv_data->Rp );
    matrix_free( cv_data->Dp );

    cv_data->Z  = NULL;
    cv_data->Rp = NULL;
    cv_data->Dp = NULL;
  }
}



bool cv_enkf_set_double( void * arg , const char * var_name , double value) {
  cv_enkf_data_type * module_data = cv_enkf_data_safe_cast( arg );
  {
    bool name_recognized = true;

    if (strcmp( var_name , ENKF_TRUNCATION_KEY_) == 0)
      cv_enkf_set_truncation( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}


bool cv_enkf_set_int( void * arg , const char * var_name , int value) {
  cv_enkf_data_type * module_data = cv_enkf_data_safe_cast( arg );
  {
    bool name_recognized = true;
    
    if (strcmp( var_name , ENKF_NCOMP_KEY_) == 0)
      cv_enkf_set_subspace_dimension( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}


bool cv_enkf_get_option( void * arg , long flag) {
  cv_enkf_data_type * cv_data = cv_enkf_data_safe_cast( arg );
  {
    return (flag & cv_data->option_flags);
  }
}




#ifdef INTERNAL_LINK
#define SYMBOL_TABLE cv_enkf_symbol_table
#else
#define SYMBOL_TABLE analysis_table
#endif

analysis_table_type SYMBOL_TABLE[] = {
  { 
    .alloc           = cv_enkf_data_alloc,
    .freef           = cv_enkf_data_free,
    .set_int         = cv_enkf_set_int , 
    .set_double      = cv_enkf_set_double , 
    .set_string      = NULL , 
    .get_option      = cv_enkf_get_option , 
    .initX           = cv_enkf_initX , 
    .init_update     = cv_enkf_init_update , 
    .complete_update = cv_enkf_complete_update
  }
};
