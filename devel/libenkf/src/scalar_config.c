#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <ens_config.h>
#include <scalar_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <trans_func.h>





scalar_config_type * scalar_config_alloc_empty(int size) {
  scalar_config_type *scalar_config = malloc(sizeof *scalar_config);

  scalar_config->data_size   	       = size;
  scalar_config->mean        	       = enkf_util_malloc(size * sizeof *scalar_config->mean        , __func__);
  scalar_config->std         	       = enkf_util_malloc(size * sizeof *scalar_config->std         ,  __func__);
  scalar_config->active      	       = enkf_util_malloc(size * sizeof *scalar_config->active      , __func__);
  scalar_config->output_transform      = enkf_util_malloc(scalar_config->data_size * sizeof * scalar_config->output_transform      , __func__);
  scalar_config->output_transform_name = enkf_util_malloc(scalar_config->data_size * sizeof * scalar_config->output_transform_name , __func__);
  scalar_config->internal_offset       = 0;
  scalar_config->void_arg              = enkf_util_malloc(scalar_config->data_size * sizeof * scalar_config->void_arg , __func__);

  {
    int i;
    for (i=0; i < size; i++) {
      scalar_config->output_transform_name[i] = NULL;
      scalar_config->void_arg[i]              = NULL;
      scalar_config->active[i]                = false;
      scalar_config->std[i]                   = 1.0;
      scalar_config->mean[i]                  = 0.0;
    }
  }
  return scalar_config;
}



void scalar_config_transform(const scalar_config_type * config , const double * input_data , double *output_data) {
  int index;
  for (index = 0; index < config->data_size; index++) {

    if (config->output_transform[index] == NULL)
      output_data[index] = input_data[index];
    else
      output_data[index] = config->output_transform[index](input_data[index] , config->void_arg[index]);
  }
}


void scalar_config_truncate(const scalar_config_type * config , double *data) {
  int i;
  for (i=0; i < config->data_size; i++) {
    if (config->active[i]) 
      if (config->output_transform[i] == NULL)
	data[i] = util_double_max(0.0 , data[i]);
  }
}




void scalar_config_fscanf_line(scalar_config_type * config , int line_nr , FILE * stream) {
  config->output_transform[line_nr] = trans_func_lookup(stream , &config->output_transform_name[line_nr] , &config->void_arg[line_nr] , &config->active[line_nr]);
}





void scalar_config_free(scalar_config_type * scalar_config) {
  int i;
  free(scalar_config->mean);
  free(scalar_config->std);
  free(scalar_config->active);
  util_free_string_list(scalar_config->output_transform_name , scalar_config->data_size);
  for (i=0; i < scalar_config->data_size; i++)
    if (scalar_config->void_arg[i] != NULL) void_arg_free(scalar_config->void_arg[i]);

  free(scalar_config->void_arg);
  free(scalar_config->output_transform);
  free(scalar_config);
}



/*****************************************************************/

GET_DATA_SIZE(scalar);
VOID_FUNC(scalar_config_free , scalar_config_type);
