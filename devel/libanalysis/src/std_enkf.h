#ifndef __SIMPLE_ENKF_H__
#define __SIMPLE_ENKF_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <matrix.h>

#define DEFAULT_ENKF_TRUNCATION_  0.99
#define DEFAULT_ENKF_ALPHA_       1.50
#define DEFAULT_ENKF_STD_CUTOFF_  1e-6

#define  ENKF_TRUNCATION_KEY_     "ENKF_TRUNCATION"
#define  ENKF_ALPHA_KEY_          "ENKF_ALPHA"
#define  STD_CUTOFF_KEY_          "STD_CUTOFF"

void     std_enkf_initX(void * module_data , matrix_type * X , matrix_type * S , matrix_type * R , matrix_type * innov , matrix_type * E , matrix_type *D);
bool     std_enkf_set_flag( void * module_data , const char * flag , int value);
bool     std_enkf_set_var( void * module_data , const char * flag , double value);




#ifdef __cplusplus
}
#endif

#endif





