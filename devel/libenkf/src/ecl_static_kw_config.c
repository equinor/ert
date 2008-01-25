#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <enkf_util.h>
#include <ecl_static_kw_config.h>
#include <enkf_macros.h>

struct ecl_static_kw_config_struct {
  int size;                 
  const char * ecl_kw_name; 
  enkf_var_type var_type;   
  char * ens_file;          
};



ecl_static_kw_config_type * ecl_static_kw_config_alloc(int size , const char * ecl_kw_name , const char * ens_file) {
  ecl_static_kw_config_type *config = malloc(sizeof *config);
  config->size          = size;
  config->ecl_kw_name   = util_alloc_string_copy(ecl_kw_name);
  config->ens_file      = util_alloc_string_copy(ens_file);
  config->var_type      = ecl_static;
  return config;
}



const char * ecl_static_kw_config_get_ensname_ref(const ecl_static_kw_config_type * ecl_static_kw_config) {
  return ecl_static_kw_config->ens_file;
}

void ecl_static_kw_config_free(ecl_static_kw_config_type *config) {
  free((char *) config->ecl_kw_name);
  free(config->ens_file);
  free(config);
}


/*****************************************************************/
VOID_CONFIG_FREE(ecl_static_kw);
/*
  GET_SERIAL_SIZE(ecl_static_kw);
  VOID_GET_SERIAL_SIZE(ecl_static_kw);

  CONFIG_SET_ECL_FILE     (ecl_static_kw);
  CONFIG_SET_ECL_FILE_VOID(ecl_static_kw);
*/

