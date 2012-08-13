#ifndef __RML_ENKF_H__
#define __RML_ENKF_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <matrix.h>
#include <rng.h>

#define  DEFAULT_ENKF_TRUNCATION_  0.99
#define  ENKF_TRUNCATION_KEY_      "ENKF_TRUNCATION"
#define  ENKF_NCOMP_KEY_           "ENKF_NCOMP" 

typedef struct rml_enkf_data_struct rml_enkf_data_type;

  void rml_enkf_updateA(void * module_data , 
			matrix_type * A , 
			matrix_type * S , 
			matrix_type * R , 
			matrix_type * dObs , 
			matrix_type * E , 
			matrix_type * D);
  
  bool     rml_enkf_set_double( void * arg , const char * var_name , double value);

  bool     rml_enkf_set_int( void * arg , const char * var_name , int value);
  int      rml_enkf_get_subspace_dimension( rml_enkf_data_type * data );
  void     rml_enkf_set_truncation( rml_enkf_data_type * data , double truncation );
  void     rml_enkf_set_subspace_dimension( rml_enkf_data_type * data , int subspace_dimension);


  double   rml_enkf_get_truncation( rml_enkf_data_type * data );
  void   * rml_enkf_data_alloc( rng_type * rng);
  void     rml_enkf_data_free( void * module_data );

  int rml_enkf_get_int( const void * arg, const char * var_name);
  bool rml_enkf_has_var( const void * arg, const char * var_name);


#ifdef __cplusplus
}
#endif

#endif





