#ifndef ERT_STD_ENKF_H
#define ERT_STD_ENKF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include <ert/res_util/matrix.hpp>
#include <ert/util/rng.hpp>

#define  DEFAULT_ENKF_TRUNCATION_  0.98
#define  ENKF_TRUNCATION_KEY_      "ENKF_TRUNCATION"
#define  ENKF_NCOMP_KEY_           "ENKF_NCOMP"
#define  USE_EE_KEY_               "USE_EE"
#define  USE_GE_KEY_               "USE_GE"
#define  ANALYSIS_SCALE_DATA_KEY_  "ANALYSIS_SCALE_DATA"

  typedef struct std_enkf_data_struct std_enkf_data_type;


  bool     std_enkf_set_double( void * arg , const char * var_name , double value);

  int      std_enkf_get_subspace_dimension( std_enkf_data_type * data );
  void     std_enkf_set_truncation( std_enkf_data_type * data , double truncation );
  void     std_enkf_set_subspace_dimension( std_enkf_data_type * data , int subspace_dimension);
  void     std_enkf_set_lambda0( std_enkf_data_type * data , double lambda0 );
  bool     std_enkf_has_var( const void * arg, const char * var_name);

  double   std_enkf_get_truncation( std_enkf_data_type * data );
  void   * std_enkf_data_alloc( );
  void     std_enkf_data_free( void * module_data );

  bool   std_enkf_get_bool( const void * arg, const char * var_name);
  int    std_enkf_get_int( const void * arg, const char * var_name);
  double std_enkf_get_double( const void * arg, const char * var_name);
  bool   std_enkf_has_var( const void * arg, const char * var_name);
  long   std_enkf_get_options( void * arg , long flag );
  bool   std_enkf_set_bool( void * arg , const char * var_name , bool value);
  bool   std_enkf_set_int( void * arg , const char * var_name , int value);
  bool   std_enkf_set_double( void * arg , const char * var_name , double value);
  void   std_enkf_initX(void * module_data ,
                        matrix_type * X ,
                        const matrix_type * A ,
                        const matrix_type * S ,
                        const matrix_type * R ,
                        const matrix_type * dObs ,
                        const matrix_type * E ,
                        const matrix_type * D,
                        rng_type * rng);

#ifdef __cplusplus
}
#endif

#endif
