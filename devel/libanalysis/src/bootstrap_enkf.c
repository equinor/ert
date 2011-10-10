/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'bootstrap_enkf.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <rng.h>
#include <matrix.h>
#include <matrix_blas.h>
#include <stdio.h>
#include <analysis_table.h>
#include <analysis_module.h>
#include <enkf_linalg.h>
#include <std_enkf.h>
#include <math.h>
#include <int_vector.h>

#define BOOTSTRAP_ENKF_TYPE_ID 741223

#define INVALID_SUBSPACE_DIMENSION  -1
#define INVALID_TRUNCATION          -1
#define DEFAULT_SUBSPACE_DIMENSION  INVALID_SUBSPACE_DIMENSION

#define  DEFAULT_DO_CV               false 
#define  DEFAULT_NFOLDS              10
#define  NFOLDS_KEY                  "BOOTSTRAP_NFOLDS"


typedef struct {
  UTIL_TYPE_ID_DECLARATION;
  rng_type             * rng; 
  long                   option_flags;
  bool                   doCV;  
} bootstrap_enkf_data_type;


static UTIL_SAFE_CAST_FUNCTION( bootstrap_enkf_data , BOOTSTRAP_ENKF_TYPE_ID )


void bootstrap_enkf_set_doCV( bootstrap_enkf_data_type * data , bool doCV) {
  data->doCV = doCV;
}


void * bootstrap_enkf_data_alloc( rng_type * rng ) {
  bootstrap_enkf_data_type * boot_data = util_malloc( sizeof * boot_data , __func__);
  UTIL_TYPE_ID_INIT( boot_data , BOOTSTRAP_ENKF_TYPE_ID );
  boot_data->option_flags = ANALYSIS_NEED_ED + ANALYSIS_UPDATE_A;
  boot_data->rng = rng;
  rng_safe_cast( rng );
  bootstrap_enkf_set_doCV( boot_data , DEFAULT_DO_CV);
  return boot_data;
}





void bootstrap_enkf_data_free( void * arg ) {
  bootstrap_enkf_data_type * boot_data = bootstrap_enkf_data_safe_cast( arg );
  {
  }
}


static int ** alloc_iens_resample( rng_type * rng , int ens_size ) {
  int ** iens_resample;
  int iens;

  iens_resample = util_malloc( ens_size * sizeof * iens_resample , __func__);
  for (iens = 0; iens < ens_size; iens++)
    iens_resample[iens] = util_malloc( ens_size * sizeof( ** iens_resample ) , __func__);
  
  {
    int i,j;
    for (i=0; i < ens_size; i++)
      for (j=0; j < ens_size; j++) 
        iens_resample[i][j] = rng_get_int( rng , ens_size );
  }
  return iens_resample;
}


static void free_iens_resample( int ** iens_resample, int ens_size ) {
  for (int i=0; i < ens_size; i++)
    free( iens_resample[i] );
  free( iens_resample );
}



void bootstrap_enkf_updateA(void * module_data , 
                            matrix_type * A , 
                            matrix_type * S , 
                            matrix_type * R , 
                            matrix_type * dObs , 
                            matrix_type * E ,
                            matrix_type * D , 
                            matrix_type * randrot) {
  
  bootstrap_enkf_data_type * bootstrap_data = bootstrap_enkf_data_safe_cast( module_data );
  {
    const int num_cpu_threads = 4;
    int ens_size              = matrix_get_columns( A );
    matrix_type * X           = matrix_alloc( ens_size , ens_size );
    matrix_type * A0          = matrix_alloc_copy( A );
    matrix_type * S_resampled = matrix_alloc_copy( S );
    matrix_type * A_resampled = matrix_alloc( matrix_get_rows(A0) , matrix_get_columns( A0 ));
    int ** iens_resample      = alloc_iens_resample( bootstrap_data->rng , ens_size );
    {
      int ensemble_members_loop;
      for ( ensemble_members_loop = 0; ensemble_members_loop < ens_size; ensemble_members_loop++) { 
        int unique_bootstrap_components;
        int ensemble_counter;
        /* Resample A and meas_data. Here we are careful to resample the working copy.*/
        printf("%d/%d \n",ensemble_members_loop , ens_size);
        {
          int_vector_type * bootstrap_components = int_vector_alloc( ens_size , 0);
          for (ensemble_counter  = 0; ensemble_counter < ens_size; ensemble_counter++) {
            int random_column = iens_resample[ ensemble_members_loop][ensemble_counter];
            int_vector_iset( bootstrap_components , ensemble_counter , random_column );
            matrix_copy_column( A_resampled , A0 , ensemble_counter , random_column );
            matrix_copy_column( S_resampled , S  , ensemble_counter , random_column );
          }
          int_vector_select_unique( bootstrap_components );
          unique_bootstrap_components = int_vector_size( bootstrap_components );
          int_vector_free( bootstrap_components );
          if (bootstrap_data->doCV) {
            //--matrix_type * U0   = matrix_alloc( nrobs , nrmin    ); /* Left singular vectors.  */
            //--matrix_type * V0T  = matrix_alloc( nrmin , ens_size ); /* Right singular vectors. */
            //--matrix_type * Z    = matrix_alloc( nrmin , nrmin    );
            //--double      * eig  = util_malloc( sizeof * eig * nrmin , __func__);
            //--enkf_analysis_invertS_pre_cv( config , S , R , V0T , Z , eig , U0 );    <-- Ikke bootstrap sensitiv == cv_enkf_init_update????
            //--enkf_analysis_allocX_pre_cv( ... );
            util_exit("%s: Sorry - bootstrap + CV not currently supported\n");
          } else {
            double truncation = 0.95;
            int ncomp = -1;
            std_enkf_initX__(X,S,R,E,D,truncation,ncomp,true);
            matrix_inplace_matmul_mt1( A_resampled , X , num_cpu_threads );
            
            matrix_inplace_add( A_resampled , A0 );
            matrix_copy_column( A , A_resampled, ensemble_members_loop, ensemble_members_loop);
          }
        }
      }
    }
    

    free_iens_resample( iens_resample , ens_size);
    matrix_free( X );
    matrix_free( S_resampled );
    matrix_free( A_resampled );
    matrix_free( A0 );
  }
}


void bootstrap_enkf_init_update( void * arg , 
                                 const matrix_type * S , 
                                 const matrix_type * R , 
                                 const matrix_type * dObs , 
                                 const matrix_type * E , 
                                 const matrix_type * D ) {
  
  bootstrap_enkf_data_type * bootstrap_data = bootstrap_enkf_data_safe_cast( arg );
  {
    
  }
}


void bootstrap_enkf_complete_update(void * arg) {
  bootstrap_enkf_data_type * bootstrap_data = bootstrap_enkf_data_safe_cast( arg );
  {
    
  }
}






long bootstrap_enkf_get_options( void * arg , long flag) {
  bootstrap_enkf_data_type * bootstrap_data = bootstrap_enkf_data_safe_cast( arg );
  {
    return bootstrap_data->option_flags;
  }
}


bool bootstrap_enkf_set_double( void * arg , const char * var_name , double value) {
  bootstrap_enkf_data_type * bootstrap_data = bootstrap_enkf_data_safe_cast( arg );
  {
  }
}


bool bootstrap_enkf_set_int( void * arg , const char * var_name , int value) {
  bootstrap_enkf_data_type * bootstrap_data = bootstrap_enkf_data_safe_cast( arg );
  {
  }
}








#ifdef INTERNAL_LINK
#define SYMBOL_TABLE bootstrap_enkf_symbol_table
#else
#define SYMBOL_TABLE analysis_table
#endif




analysis_table_type SYMBOL_TABLE[] = {
  { 
    .alloc           = bootstrap_enkf_data_alloc,
    .freef           = bootstrap_enkf_data_free,
    .set_int         = bootstrap_enkf_set_int , 
    .set_double      = bootstrap_enkf_set_double , 
    .set_bool        = NULL , 
    .set_string      = NULL , 
    .get_options     = bootstrap_enkf_get_options , 
    .initX           = NULL,
    .updateA         = bootstrap_enkf_updateA , 
    .init_update     = bootstrap_enkf_init_update , 
    .complete_update = bootstrap_enkf_complete_update
  }
};
