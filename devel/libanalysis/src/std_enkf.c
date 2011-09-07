#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <matrix.h>
#include <stdio.h>
#include <analysis_table.h>
#include <enkf_linalg.h>
#include <std_enkf.h>

#define STD_ENKF_TYPE_ID 261123



typedef struct {
  UTIL_TYPE_ID_DECLARATION;
  double    truncation;
  double    overlap_alpha;
  double    std_cutoff;
} std_enkf_data_type;



static UTIL_SAFE_CAST_FUNCTION( std_enkf_data , STD_ENKF_TYPE_ID )


void std_enkf_set_truncation( std_enkf_data_type * data , double truncation ) {
  data->truncation = truncation;
}

void std_enkf_set_alpha( std_enkf_data_type * data , double alpha) {
  data->overlap_alpha = alpha;
}

void std_enkf_set_cutoff( std_enkf_data_type * data , double cutoff) {
  data->std_cutoff = cutoff;
}

void * std_enkf_alloc( ) {
  std_enkf_data_type * data = util_malloc( sizeof * data , __func__ );
  UTIL_TYPE_ID_INIT( data , STD_ENKF_TYPE_ID );

  std_enkf_set_truncation( data , DEFAULT_ENKF_TRUNCATION_ );
  std_enkf_set_alpha( data , DEFAULT_ENKF_ALPHA_ );
  std_enkf_set_cutoff( data , DEFAULT_ENKF_STD_CUTOFF_ );
  
  return data;
}


void std_enkf_free( void * data ) { 
  free( data );
}



void std_enkf_initX(void * module_data , matrix_type * X , matrix_type * S , matrix_type * R , matrix_type * innov , matrix_type * E , matrix_type *D) {
  double truncation = 0.99;
  int ncomp         = -1;
  int nrobs         = matrix_get_rows( S );
  int ens_size      = matrix_get_columns( S );
  int nrmin         = util_int_min( ens_size , nrobs); 
  
  matrix_type * W   = matrix_alloc(nrobs , nrmin);                      
  double      * eig = util_malloc( sizeof * eig * nrmin , __func__);    

  enkf_linalg_lowrankCinv( S , R , W , eig , truncation , ncomp);    
  matrix_diag_set_scalar( X , 1.0 );

  matrix_free( W );
  free( eig );
}


bool std_enkf_set_int( void * module_data , const char * flag , int value) {
  return true;
}




bool std_enkf_set_double( void * arg , const char * var_name , double value) {
  std_enkf_data_type * module_data = std_enkf_data_safe_cast( arg );
  {
    bool name_recognized = true;

    if (strcmp( var_name , ENKF_TRUNCATION_KEY_) == 0)
      std_enkf_set_truncation( module_data , value );
    else if (strcmp( var_name , ENKF_ALPHA_KEY_) == 0)
      std_enkf_set_alpha( module_data , value );
    else if (strcmp( var_name , STD_CUTOFF_KEY_) == 0)
      std_enkf_set_cutoff( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}




/**
   gcc -fpic -c <object_file> -I??  <src_file>
   gcc -shared -o <lib_file> <object_files>
*/


#define INTERNAL_LINK

#ifdef INTERNAL_LINK
#define SYMBOL_TABLE std_enkf_symbol_table
#else
#define SYMBOL_TABLE analysis_table
#endif

analysis_table_type SYMBOL_TABLE[] = {
  { 
    .name       = "std_enkf" , 
    .alloc      = std_enkf_alloc,
    .freef      = std_enkf_free,
    .set_int    = std_enkf_set_int , 
    .set_double = std_enkf_set_double , 
    .set_string = NULL , 
    .initX      = std_enkf_initX , 
  },
  { 
    .name = NULL 
  }
};

