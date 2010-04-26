#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <scalar_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <trans_func.h>
#include <active_list.h>

#define SCALAR_CONFIG_TYPE_ID 877065

struct scalar_config_struct {
  UTIL_TYPE_ID_DECLARATION;
  int 		     data_size;
  int 		     internal_offset;
  active_list_type * active_list;

  trans_func_type ** transform;  
};




scalar_config_type * scalar_config_alloc_empty(int size) {
  scalar_config_type *scalar_config    = util_malloc(sizeof *scalar_config, __func__);
  UTIL_TYPE_ID_INIT( scalar_config , SCALAR_CONFIG_TYPE_ID );
  scalar_config->data_size   	       = size;
  scalar_config->active_list           = active_list_alloc( ALL_ACTIVE );
  scalar_config->internal_offset       = 0;
  
  scalar_config->transform             = util_malloc(scalar_config->data_size * sizeof * scalar_config->transform             , __func__);
  return scalar_config;
}



void scalar_config_transform(const scalar_config_type * config , const double * input_data , double *output_data) {
  int index;
  for (index = 0; index < config->data_size; index++) 
    output_data[index] = trans_func_eval( config->transform[index] , input_data[index] );
}

 

void scalar_config_truncate(const scalar_config_type * config , double *data) {
  return;
}




void scalar_config_fscanf_line(scalar_config_type * config , int line_nr , FILE * stream) {
  config->transform[line_nr] = trans_func_fscanf_alloc( stream );
}



void scalar_config_free(scalar_config_type * scalar_config) {
  int i;
  active_list_free(scalar_config->active_list);
  for (i=0; i < scalar_config->data_size; i++) 
    trans_func_free( scalar_config->transform[i] );
                     
  util_safe_free( scalar_config->transform );
  free(scalar_config);
}



/*****************************************************************/

SAFE_CAST(scalar_config , SCALAR_CONFIG_TYPE_ID)
GET_DATA_SIZE(scalar);
GET_ACTIVE_LIST(scalar);
VOID_FREE(scalar_config);
