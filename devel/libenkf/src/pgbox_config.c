#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <util.h>
#include <ecl_util.h>
#include <ecl_box.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <ecl_kw.h>
#include <ecl_grid.h>
#include <pgbox_config.h>
#include <field_config.h>
#include <field.h>
#include <void_arg.h>



/* 
   Input i1,i2,.... is in the ECLIPSE domain - i.e. with offset 1.
*/
pgbox_config_type * pgbox_config_alloc(const ecl_grid_type * grid , 
				       pgfilter_ftype      * filter,
				       const void_arg_type * arg,
				       int stride , 
				       int i1 , int i2 , int j1 , int j2 , int k1 , int k2, field_config_type * target_config) {

  {
    ecl_type_enum target_type = field_config_get_ecl_type(target_config);
    if (!((target_type == ecl_float_type) || (target_type == ecl_double_type))) {
      fprintf(stderr,"%s: sorry target field must have data type ecl_double_type / ecl_float_type - aborting. \n",__func__);
      abort();
    }
  }
  {
    pgbox_config_type * config = malloc(sizeof * config);
    config->stride           = stride;
    config->filter           = filter;
    config->arg              = arg;
    config->ecl_kw_name      = NULL;
    config->write_compressed = true;
    config->target_key       = util_alloc_string_copy(field_config_get_ecl_kw_name(target_config));
    {
      int nx,ny,nz,elements;
      ecl_box_type * box;
      ecl_grid_get_dims(grid , &nx , &ny , &nz , NULL);
      box = ecl_box_alloc(nx , ny , nz , i1, i2 , j1 , j2 , k1 , k2);
      
      elements           = ecl_grid_count_box_active(grid , box);
      config->data_size  = elements * config->stride;
      config->index_list = util_malloc(elements * sizeof * config->index_list , __func__);
      ecl_grid_set_box_active_list(grid , box , config->index_list);
      ecl_box_free(box);
    }
    return config;
  }
}

int pgbox_config_get_byte_size(const pgbox_config_type * config) {
  return config->data_size * ecl_util_get_sizeof_ctype(ecl_double_type);
}


void pgbox_config_apply(const pgbox_config_type * config , const double * data , field_type * target_field) {
  const int elements     = config->data_size / config->stride;
  double * target_buffer = util_malloc(elements * ecl_util_get_sizeof_ctype(ecl_double_type)  , __func__);
  int index;
  
  for (index = 0; index < elements; index++) 
    target_buffer[index] = config->filter(&data[index * config->stride] , config->arg);
  
  field_indexed_set(target_field , ecl_double_type , elements , config->index_list , target_buffer);
  free(target_buffer);
}


const char * pgbox_config_get_target_key(const pgbox_config_type * config) {
  return config->target_key;
}

void pgbox_config_free(pgbox_config_type * config) {
  free(config->index_list);
  if (config->ecl_kw_name != NULL) free(config->ecl_kw_name);
  free(config->target_key);
  free(config);
}


bool pgbox_config_write_compressed(const pgbox_config_type * config) {
  return config->write_compressed;
}


/*****************************************************************/

VOID_FREE_CONFIG(pgbox);
GET_DATA_SIZE(pgbox);
