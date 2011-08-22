#include <stdlib.h>
#include <matrix.h>
#include <stdio.h>
#include <analysis_table.h>
#include <enkf_linalg.h>

typedef struct {
  double param1;
  double param2;
} simple_enkf_data_type;


void * simple_enkf_alloc( ) {
  simple_enkf_data_type * data = util_malloc( sizeof * data , __func__ );
  return data;
}


void simple_enkf_free( void * data ) { 
  free( data );
}



static void initX(void * module_data , matrix_type * X , matrix_type * S , matrix_type * R , matrix_type * innov , matrix_type * E , matrix_type *D) {
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


bool set_flag( void * module_data , const char * flag , int value) {
  return true;
}

bool set_string( void * module_data , const char * var , const char * value) {
  return true;
}


bool set_var( void * module_data , const char * flag , double value) {
  return true;
}



/**
   gcc -fpic -c <object_file> -I??  <src_file>
   gcc -shared -o <lib_file> <object_files>
*/


#define INTERNAL_LINK

#ifdef INTERNAL_LINK
#define SYMBOL_TABLE simple_enkf_symbol_table
#else
#define SYMBOL_TABLE analysis_table
#endif

analysis_table_type SYMBOL_TABLE[] = {
  { 
    .name       = "SimpleEnKF" , 
    .alloc      = simple_enkf_alloc,
    .freef      = simple_enkf_free,
    .set_flag   = set_flag , 
    .set_var    = set_var , 
    .set_string = set_string,
    .initX      = initX , 
  },
  { 
    .name = NULL 
  }
};

