#ifndef __RML_ENKF_COMMON_H__
#define __RML_ENKF_COMMON_H__

#include <stdbool.h>

#include <ert/util/matrix.h>
#include <ert/util/rng.h>
#include <ert/util/bool_vector.h>



void rml_enkf_common_initA__( matrix_type * A ,
                              matrix_type * S , 
                              matrix_type * Cd , 
                              matrix_type * E , 
                              matrix_type * D ,
                              double truncation,
                              double lamda,
                              matrix_type * Ud,
                              double * Wd,
                              matrix_type * VdT);


void rml_enkf_common_store_state( matrix_type * state , const matrix_type * A , const bool_vector_type * ens_mask );
void rml_enkf_common_recover_state( const matrix_type * state , matrix_type * A , const bool_vector_type * ens_mask );

#endif
