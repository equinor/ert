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

/*****************************************************************/

struct summary_struct {
  int                          __type_id; /* Only used for run_time checking. */
  const summary_config_type  * config;
  double                     * data;         /* Size is always one - but what the fuck ... */
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
  summary->config = summary_config;
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


void summary_fread(summary_type * summary , FILE * stream) {
  int  size;
  enkf_util_fread_assert_target_type(stream , SUMMARY);
  fread(&size , sizeof  size , 1 , stream);
  util_fread(summary->data , sizeof *summary->data , size , stream , __func__);
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




bool summary_fwrite(const summary_type * summary , FILE * stream , bool internal_state) {
  const  summary_config_type * config = summary->config;
  const int data_size = summary_config_get_data_size(config);
  
  enkf_util_fwrite_target_type(stream , SUMMARY);
  fwrite(&data_size , sizeof  data_size     , 1 , stream);
  util_fwrite(summary->data  , sizeof *summary->data    ,data_size , stream , __func__);
  return true;
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




void summary_ecl_load(summary_type * summary , const char * ecl_file_name , const ecl_sum_type * ecl_sum, const ecl_file_type * ecl_file , int report_step) {
  if (ecl_sum != NULL) {
    const char * var_key               = summary_config_get_var(summary->config);
    const ecl_smspec_var_type var_type = summary_config_get_var_type(summary->config);
    int ministep2;
    ecl_sum_report2ministep_range(ecl_sum , report_step , NULL , &ministep2);
    if ((var_type == ECL_SMSPEC_WELL_VAR) || (var_type == ECL_SMSPEC_GROUP_VAR)) {
      /* .. check if the/group well is defined in the smspec file (i.e. if it is open). */
      if (ecl_sum_has_general_var(ecl_sum , var_key))
	summary->data[0] = ecl_sum_get_general_var(ecl_sum , ministep2  , var_key);
      else 
	/* 
	   The summary object does not have this well/group - probably
	   meaning that it has not yet opened. If the user has
	   mis-spelled the name, we will go through the whole
	   simulation without detecting that error. 
	*/
	summary->data[0] = 0;
    } else {
      if (!ecl_sum_has_general_var(ecl_sum , var_key))
	util_abort("%s: sorry - could not find variable: \"%s\" in summary files. \n",__func__ , var_key);
      summary->data[0] = ecl_sum_get_general_var(ecl_sum , ministep2  ,var_key );
    }
  }
}


void summary_ensemble_fprintf_results(const summary_type ** ensemble, int ens_size, const char * filename)
{
  int        * index = util_malloc(ens_size * sizeof * index , __func__);
  const char * var   = summary_config_get_var(ensemble[0]->config);
  double     * data  = util_malloc(ens_size * sizeof * data , __func__);

  for(int i=0; i<ens_size; i++) {
    data[i]  = *(ensemble[i]->data);
    index[i] = i;
  }
  
  {
    FILE * stream = util_fopen(filename , "w");
    enkf_util_fprintf_data( index , (const double ** ) &data, "Member #" , &var, ens_size, 1, true, stream);
    fclose(stream);
  }
  
  free(index);
  free(data);
}





/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/
SAFE_CAST(summary , SUMMARY)
MATH_OPS(summary)
VOID_ALLOC(summary)
VOID_FREE(summary)
VOID_FREE_DATA(summary)
VOID_REALLOC_DATA(summary)
VOID_FWRITE (summary)
VOID_FREAD  (summary)
VOID_COPYC     (summary)
VOID_SERIALIZE(summary)
VOID_DESERIALIZE(summary)
VOID_ECL_LOAD(summary)
VOID_USER_GET(summary)
VOID_FPRINTF_RESULTS(summary)
VOID_STORE(summary)
VOID_LOAD(summary)
VOID_MATRIX_SERIALIZE(summary)
VOID_MATRIX_DESERIALIZE(summary)
