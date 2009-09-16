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

/*****************************************************************/

struct summary_struct {
  int                          __type_id;    /* Only used for run_time checking. */
  summary_config_type        * config;       /* Can not be NULL - var_type is set on first load. */
  double                     * data;         /* Size is always one - but what the fuck ... */
  summary_type               * min_variance;
};




void summary_clear(summary_type * summary) {
  const int size = summary_config_get_data_size(summary->config);   
  int k;
  for (k = 0; k < size; k++)
    summary->data[k] = 0.0;
}


void summary_realloc_data(summary_type *summary) {
  summary->data = util_malloc(summary_config_get_data_size(summary->config) * sizeof *summary->data , __func__);
}


void summary_free_data(summary_type *summary) {
  free(summary->data);
  summary->data = NULL;
}


summary_type * summary_alloc(const summary_config_type * summary_config) {
  summary_type * summary  = util_malloc(sizeof *summary , __func__);
  summary->__type_id      = SUMMARY;
  summary->config = (summary_config_type *) summary_config;
  summary->data = NULL;
  summary_realloc_data(summary);
  return summary;
}




summary_type * summary_copyc(const summary_type *summary) {
  const int size = summary_config_get_data_size(summary->config);   
  summary_type * new = summary_alloc(summary->config);
  
  memcpy(new->data , summary->data , size * sizeof *summary->data);
  return new;
}




void summary_load(summary_type * summary , buffer_type * buffer) {
  int  size = summary_config_get_data_size( summary->config );
  enkf_util_assert_buffer_type( buffer , SUMMARY );
  buffer_fread( buffer , summary->data , sizeof * summary->data , size);
}



void summary_upgrade_103( const char * filename ) {
  FILE * stream            = util_fopen(filename , "r");
  enkf_impl_type impl_type = util_fread_int( stream );
  int            size      = util_fread_int( stream );
  double * data            = util_malloc( size * sizeof * data , __func__);
  util_fread(data , sizeof * data , size , stream , __func__);
  fclose(stream);
  
  {
    buffer_type * buffer = buffer_alloc( 100 );
    buffer_fwrite_time_t( buffer , time(NULL));
    buffer_fwrite_int( buffer , impl_type );
    buffer_fwrite( buffer , data , sizeof * data , size);
    buffer_store( buffer , filename );
    buffer_free( buffer );
  }
  free( data );
}





bool summary_store(const summary_type * summary , buffer_type * buffer, bool internal_state) {
  int  size = summary_config_get_data_size( summary->config );
  buffer_fwrite_int( buffer , SUMMARY );
  buffer_fwrite( buffer , summary->data , sizeof * summary->data , size);
  return true;
}


void summary_free(summary_type *summary) {
  summary_free_data(summary);
  free(summary);
}




void summary_deserialize(const summary_type * summary , serial_state_type * serial_state , const serial_vector_type * serial_vector) {
  const summary_config_type *config      = summary->config;
  const int                data_size   = summary_config_get_data_size(config);
  const active_list_type  *active_list = summary_config_get_active_list(config);
  enkf_deserialize(summary->data , data_size , ecl_double_type , active_list , serial_state , serial_vector);
}


int summary_serialize(const summary_type *summary , serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  const summary_config_type *config    = summary->config;
  const int                data_size   = summary_config_get_data_size(config);
  const active_list_type  *active_list = summary_config_get_active_list(config);
  
  return enkf_serialize(summary->data , data_size , ecl_double_type , active_list , serial_state , serial_offset , serial_vector);
}


void summary_matrix_serialize(const summary_type * summary , const active_list_type * active_list , matrix_type * A , int row_offset , int column) {
  const summary_config_type *config  = summary->config;
  const int                data_size = summary_config_get_data_size(config);
  
  enkf_matrix_serialize( summary->data , data_size , ecl_double_type , active_list , A , row_offset , column);
}


void summary_matrix_deserialize(summary_type * summary , const active_list_type * active_list , const matrix_type * A , int row_offset , int column) {
  const summary_config_type *config  = summary->config;
  const int                data_size = summary_config_get_data_size(config);
  
  enkf_matrix_deserialize( summary->data , data_size , ecl_double_type , active_list , A , row_offset , column);
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
      int ministep2;
      ecl_sum_report2ministep_range(ecl_sum , report_step , NULL , &ministep2);

      if ((var_type == ECL_SMSPEC_WELL_VAR) || (var_type == ECL_SMSPEC_GROUP_VAR)) {
        /* .. check if the/group well is defined in the smspec file (i.e. if it is open). */
        if (ecl_sum_has_general_var(ecl_sum , var_key)) 
          summary->data[0] = ecl_sum_get_general_var(ecl_sum , ministep2  , var_key);
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
        summary->data[0] = ecl_sum_get_general_var(ecl_sum , ministep2  ,var_key );
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
  
  return loadOK;
}



void summary_set_inflation(summary_type * inflation , const summary_type * std , const summary_type * min_std, log_type * logh) {
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
SAFE_CAST(summary , SUMMARY)
SAFE_CONST_CAST(summary , SUMMARY)
VOID_ALLOC(summary)
VOID_FREE(summary)
VOID_FREE_DATA(summary)
VOID_REALLOC_DATA(summary)
VOID_COPYC     (summary)
VOID_SERIALIZE(summary)
VOID_DESERIALIZE(summary)
VOID_ECL_LOAD(summary)
VOID_USER_GET(summary)
VOID_STORE(summary)
VOID_LOAD(summary)
VOID_MATRIX_SERIALIZE(summary)
VOID_MATRIX_DESERIALIZE(summary)
VOID_SET_INFLATION(summary)
VOID_CLEAR(summary)
VOID_IADD(summary)
VOID_SCALE(summary)
VOID_IMUL(summary)
VOID_IADDSQR(summary)
VOID_ISQRT(summary)
