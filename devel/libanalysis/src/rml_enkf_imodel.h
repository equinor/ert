#ifndef __RML_ENKF_IMODEL_H__
#define __RML_ENKF_IMODEL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <matrix.h>
#include <rng.h>

#define  DEFAULT_ENKF_TRUNCATION_  0.99
#define  ENKF_TRUNCATION_KEY_      "ENKF_TRUNCATION"
#define  ENKF_NCOMP_KEY_           "ENKF_NCOMP" 

typedef struct rml_enkf_imodel_data_struct rml_enkf_imodel_data_type;



  void     rml_enkf_imodel_Create_Csc(rml_enkf_imodel_data_type * data);
  void     rml_enkf_imodel_updateA(void * module_data ,  matrix_type * A , matrix_type * S , matrix_type * R , matrix_type * dobs , matrix_type * E , matrix_type *D );
  bool     rml_enkf_imodel_set_double( void * arg , const char * var_name , double value);
  
  bool     rml_enkf_imodel_set_int( void * arg , const char * var_name , int value);
  int      rml_enkf_imodel_get_subspace_dimension( rml_enkf_imodel_data_type * data );
  void     rml_enkf_imodel_set_truncation( rml_enkf_imodel_data_type * data , double truncation );
  void     rml_enkf_imodel_set_subspace_dimension( rml_enkf_imodel_data_type * data , int subspace_dimension);
  
  
  double   rml_enkf_imodel_get_truncation( rml_enkf_imodel_data_type * data );
  void   * rml_enkf_imodel_data_alloc( rng_type * rng);
  void     rml_enkf_imodel_data_free( void * module_data );
  bool     rml_enkf_imodel_has_var( const void * arg, const char * var_name);
  
  void     rml_enkf_imodel_init1__( matrix_type * A,
				    rml_enkf_imodel_data_type * data, 
				    double truncation,
				    double nsc);
  
  void     rml_enkf_imodel_init2__( rml_enkf_imodel_data_type * data,
				    matrix_type *A,
				    matrix_type *Acopy,
				    double * Wdr,
				    double nsc,
				    matrix_type * VdTr);
  
  void     rml_enkf_imodel_scalingA(matrix_type *A, double * Csc, bool invert);
  int      rml_enkf_imodel_get_int( const void * arg, const char * var_name);
  
#ifdef __cplusplus
}
#endif

#endif





