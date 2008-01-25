#include <stdlib.h>
#include <util.h>
#include <enkf_util.h>
#include <enkf_ecl_kw_config.h>


enkf_ecl_kw_config_type * enkf_ecl_kw_config_alloc(int size , const char *ecl_kw_name , const char * ensfile) {
  enkf_ecl_kw_config_type *config = malloc(sizeof *config);
  config->size          = size;
  config->ecl_kw_name   = util_alloc_string_copy(ecl_kw_name);
  config->ensfile      = util_alloc_string_copy(ensfile);
  config->var_type      = ecl_restart;
  return config;
}


enkf_ecl_kw_config_type * enkf_ecl_kw_config_copyc(const enkf_ecl_kw_config_type * src) {
  enkf_ecl_kw_config_type * new = enkf_ecl_kw_config_alloc(src->size , src->ecl_kw_name , src->ensfile);
  return new;
}


const char * enkf_ecl_kw_config_get_ensfile_ref(const enkf_ecl_kw_config_type * enkf_ecl_kw_config) {
  return enkf_ecl_kw_config->ensfile;
}


void enkf_ecl_kw_config_free(enkf_ecl_kw_config_type *config) {
  free((char *) config->ecl_kw_name);
  free(config->ensfile);
  free(config);
}

/*****************************************************************/

VOID_FREE(enkf_ecl_kw_config);
GET_SIZE(enkf_ecl_kw);
VOID_GET_SIZE(enkf_ecl_kw);
