#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <enkf_macros.h>
#include <enkf_util.h>
#include <util.h>
#include <summary.h>
#include <summary_config.h>
#include <ecl_sum.h>
#include <enkf_types.h>
#include <enkf_util.h>
#include <enkf_serialize.h>

#define  DEBUG
#define  TARGET_TYPE SUMMARY
#include "enkf_debug.h"


/*****************************************************************/

struct summary_struct {
  DEBUG_DECLARE
  const summary_config_type * config;
  double *data;    /* Size is always one - but what the fuck ... */
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
  summary_type * summary  = malloc(sizeof *summary);
  summary->config = summary_config;
  summary->data = NULL;
  summary_realloc_data(summary);
  DEBUG_ASSIGN(summary)
  return summary;
}




summary_type * summary_copyc(const summary_type *summary) {
  const int size = summary_config_get_data_size(summary->config);   
  summary_type * new = summary_alloc(summary->config);
  
  memcpy(new->data , summary->data , size * sizeof *summary->data);
  return new;
}


void summary_fread(summary_type * summary , FILE * stream) {
  DEBUG_ASSERT(summary); 
  {
    int  size;
    enkf_util_fread_assert_target_type(stream , SUMMARY);
    fread(&size , sizeof  size , 1 , stream);
    enkf_util_fread(summary->data , sizeof *summary->data , size , stream , __func__);
  }
}



bool summary_fwrite(const summary_type * summary , FILE * stream) {
  DEBUG_ASSERT(summary); 
  {
    const  summary_config_type * config = summary->config;
    const int data_size = summary_config_get_data_size(config);
    
    enkf_util_fwrite_target_type(stream , SUMMARY);
    fwrite(&data_size , sizeof  data_size     , 1 , stream);
    enkf_util_fwrite(summary->data  , sizeof *summary->data    ,data_size , stream , __func__);
  }
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
  const summary_config_type *config      = summary->config;
  const int                data_size   = summary_config_get_data_size(config);
  const active_list_type  *active_list = summary_config_get_active_list(config);

  return enkf_serialize(summary->data , data_size , ecl_double_type , active_list , serial_state , serial_offset , serial_vector);
}


double summary_get(const summary_type * summary) {
  DEBUG_ASSERT(summary)
  {
    return summary->data[0]; /* That is all it has got ... */
  }
}


double summary_user_get(const summary_type * summary , const char * index_key , bool * valid) {
  DEBUG_ASSERT(summary)
  *valid = true;
  return summary->data[0];
}




void summary_ecl_load(summary_type * summary , const char * ecl_file , const ecl_sum_type * ecl_sum, const ecl_block_type * ecl_block , int report_step) {
  DEBUG_ASSERT(summary)
  if (ecl_sum != NULL) {
    const char * var_key            = summary_config_get_var(summary->config);
    const ecl_sum_var_type var_type = summary_config_get_var_type(summary->config);
    if (var_type == ecl_sum_well_var) {
      /* .. check if the well is defined in the smspec file (i.e. if it is open). */
      if (ecl_sum_has_general_var(ecl_sum , var_key))
	summary->data[0] = ecl_sum_get_general_var(ecl_sum , report_step  , var_key);
      else 
	/* 
	   The summary object does not have this well - probably
	   meaning that it has not yet opened. If the user has
	   mis-spelled the name, we will go through the whole
	   simulation without detecting that error.
	*/
	summary->data[0] = 0;
    } else
      summary->data[0] = ecl_sum_get_general_var(ecl_sum , report_step  ,var_key );
  }
}



/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/
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
