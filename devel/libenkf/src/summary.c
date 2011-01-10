#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <enkf_macros.h>
#include <enkf_util.h>
#include <util.h>
#include <summary.h>
#include <summary_config.h>
#include <ecl_sum.h>
#include <ecl_smspec.h> 
#include <ecl_file.h>
#include <enkf_types.h>
#include <enkf_util.h>
#include <enkf_serialize.h>
#include <log.h>
#include <double_vector.h>

/*****************************************************************/

#define SUMMARY_UNDEF -9999

struct summary_struct {
  int                          __type_id;     /* Only used for run_time checking. */
  summary_config_type        * config;        /* Can not be NULL - var_type is set on first load. */
  double                     * data;          /* Size is always one - but what the fuck ... */
  double_vector_type         * data_ts;       /* Time series of the summary data - NOT in use. */ 
};




void summary_clear(summary_type * summary) {
  const int data_size = summary_config_get_data_size( summary->config );
  for (int k=0; k < data_size; k++)
    summary->data[k] = 0;
}



summary_type * summary_alloc(const summary_config_type * summary_config) {
  summary_type * summary  = util_malloc(sizeof *summary , __func__);
  summary->__type_id      = SUMMARY;
  summary->config         = (summary_config_type *) summary_config;
  summary->data_ts        = double_vector_alloc(0 , SUMMARY_UNDEF);  
  {
    const int data_size = summary_config_get_data_size( summary_config );
    summary->data       = util_malloc( data_size * sizeof * summary->data , __func__ );
  }
  return summary;
}




void summary_copy(const summary_type *src , summary_type * target) {
  if (src->config == target->config) {
    const int data_size = summary_config_get_data_size( src->config );
    for (int k=0; k < data_size; k++)
      target->data[k] = src->data[k];
    
    double_vector_memcpy( target->data_ts , src->data_ts );
  } else
    util_abort("%s: do not share config objects \n",__func__);
}




void summary_load(summary_type * summary , buffer_type * buffer, int report_step) {
  int  size = summary_config_get_data_size( summary->config );
  enkf_util_assert_buffer_type( buffer , SUMMARY );
  buffer_fread( buffer , summary->data , sizeof * summary->data , size);
  double_vector_iset( summary->data_ts , report_step , summary->data[0] );
}






bool summary_store(const summary_type * summary , buffer_type * buffer, int report_step , bool internal_state) {
  int  size = summary_config_get_data_size( summary->config );
  buffer_fwrite_int( buffer , SUMMARY );
  buffer_fwrite( buffer , summary->data , sizeof * summary->data , size);
  return true;
}


void summary_free(summary_type *summary) {
  double_vector_free( summary->data_ts );
  free(summary->data);
  free(summary);
}




void summary_serialize(const summary_type * summary , const active_list_type * active_list , matrix_type * A , int row_offset , int column) {
  const summary_config_type *config  = summary->config;
  const int                data_size = summary_config_get_data_size(config );
  
  enkf_matrix_serialize( summary->data , data_size , ECL_DOUBLE_TYPE , active_list , A , row_offset , column);
}



void summary_deserialize(summary_type * summary , const active_list_type * active_list , const matrix_type * A , int row_offset , int column) {
  const summary_config_type *config  = summary->config;
  const int                data_size = summary_config_get_data_size(config );
  
  enkf_matrix_deserialize( summary->data , data_size , ECL_DOUBLE_TYPE , active_list , A , row_offset , column);
}




double summary_get(const summary_type * summary) {
  return summary->data[0]; /* That is all it has got ... */
}


double summary_user_get(const summary_type * summary , const char * index_key , bool * valid) {
  *valid = true;
  return summary->data[0];
}



/**
   There are three typical reasons why the node data can not be loaded:

     1. The ecl_sum instance is equal to NULL.
     2. The ecl_sum instance does not have the report step we are asking for.
     3. The ecl_sum instance does not have the variable we are asking for.

   In the two first cases the function will return false, ultimately
   signaling that the simulation has failed. In the last case we
   return true, because this is a typical situation for e.g. a well
   which has not yet opened.
*/

bool summary_ecl_load(summary_type * summary , const char * ecl_file_name , const ecl_sum_type * ecl_sum, const ecl_file_type * ecl_file , int report_step) {
  bool loadOK = false;
  if (ecl_sum != NULL) {
    const char * var_key               = summary_config_get_var(summary->config);
    const ecl_smspec_var_type var_type = summary_config_get_var_type(summary->config , ecl_sum);
    
    /* Check if the ecl_sum instance has this report step. */
    if (ecl_sum_has_report_step( ecl_sum , report_step )) {
      int last_report_index = ecl_sum_iget_report_end( ecl_sum , report_step );

      if ((var_type == ECL_SMSPEC_WELL_VAR) || (var_type == ECL_SMSPEC_GROUP_VAR)) {
        /* .. check if the/group well is defined in the smspec file (i.e. if it is open). */
        if (ecl_sum_has_general_var(ecl_sum , var_key)) 
          summary->data[0] = ecl_sum_iget_general_var(ecl_sum , last_report_index  , var_key);
        else 
          /* 
           The summary object does not have this well/group - probably
           meaning that it has not yet opened. We return loadOK ==
           true in this case.
           
           If the user has misspelled the name, we will go through
           the whole simulation without detecting that error.
        */
          summary->data[0] = 0;
        loadOK = true;   
      } else if (ecl_sum_has_general_var(ecl_sum , var_key)) {
        summary->data[0] = ecl_sum_iget_general_var(ecl_sum , last_report_index  ,var_key );
        loadOK = true;
      }
    } else if (report_step == 0) {
      summary->data[0] = 0;
      loadOK = true;  
      /* 
         We do not signal load failure if we do not have the S0000
         summary file - which does not contain any useful information
         anyway. 
         
         Hmmm - there is a "if (report_step > 0)" check in the
         enkf_state_internalize_x() function as well.
      */
    } 
  } 
  if (loadOK)
    double_vector_iset( summary->data_ts , report_step , summary->data[0] );
  return loadOK;
}



void summary_set_inflation(summary_type * inflation , const summary_type * std , const summary_type * min_std) {
  int size = 1;
  for (int i = 0; i < size; i++) 
    inflation->data[i] = util_double_max( 1.0 , min_std->data[i] / std->data[i]);
}


void summary_iadd( summary_type * summary , const summary_type * delta) {
  int size = 1;
  for (int i = 0; i < size; i++) 
    summary->data[i] += delta->data[i];
}


void summary_iaddsqr( summary_type * summary , const summary_type * delta) {
  int size = 1;
  for (int i = 0; i < size; i++) 
    summary->data[i] += delta->data[i] * delta->data[i];
}


void summary_imul( summary_type * summary , const summary_type * delta) {
  int size = 1;
  for (int i = 0; i < size; i++) 
    summary->data[i] *= delta->data[i];
}

void summary_scale( summary_type * summary , double scale_factor) {
  int size = 1;
  for (int i = 0; i < size; i++) 
    summary->data[i] *= scale_factor;
}

void summary_isqrt( summary_type * summary ) {
  int size = 1;
  for (int i = 0; i < size; i++) 
    summary->data[i] = sqrt( summary->data[i] );
}





/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/
UTIL_SAFE_CAST_FUNCTION(summary , SUMMARY)
UTIL_SAFE_CAST_FUNCTION_CONST(summary , SUMMARY)
VOID_ALLOC(summary)
VOID_FREE(summary)
VOID_COPY     (summary)
VOID_ECL_LOAD(summary)
VOID_USER_GET(summary)
VOID_STORE(summary)
VOID_LOAD(summary)
VOID_SERIALIZE(summary)
VOID_DESERIALIZE(summary)
VOID_SET_INFLATION(summary)
VOID_CLEAR(summary)
VOID_IADD(summary)
VOID_SCALE(summary)
VOID_IMUL(summary)
VOID_IADDSQR(summary)
VOID_ISQRT(summary)
     
