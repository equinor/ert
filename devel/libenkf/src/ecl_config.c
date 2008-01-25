#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <util.h>
#include <hash.h>
#include <ecl_config.h>

struct ecl_config_struct {
  char 		   *root_path;
  hash_type        *enkf_ecl_kw;  
  /* A list of the eclipse keywords which will be used in the EnKF analysis - i.e pressur and saturations.*/
};


/*****************************************************************/



bool ecl_config_restart_kw(const ecl_config_type *ecl_config, const char * ecl_kw_name) {
  if (hash_has_key(ecl_config->enkf_ecl_kw , ecl_kw_name))
    return true;
  else
    return false;
}


void ecl_config_add_enkf_kw(const ecl_config_type *ecl_config, const char *ecl_kw_name) {
  hash_insert_int(ecl_config->enkf_ecl_kw , ecl_kw_name , 1); 
}
								 

void ecl_config_set_root_path(ecl_config_type * ecl_config , const char * root_path) {

  if (root_path == NULL) {
    fprintf(stderr,"%s: root_path == NULL - invalid - aborting \n",__func__);
    abort();
  }
  ecl_config->root_path = util_realloc_string_copy(ecl_config->root_path , root_path);
  util_make_path(ecl_config->root_path);
}



ecl_config_type * ecl_config_alloc(const char * root_path) {
  ecl_config_type * ecl_config = malloc(sizeof *ecl_config);
  ecl_config->root_path = NULL;
  ecl_config->enkf_ecl_kw = hash_alloc(10);
  ecl_config_set_root_path(ecl_config , root_path );
  return ecl_config;
}


char * ecl_config_alloc_eclname(const ecl_config_type * ecl_config, const char * ext_name) {
  char * ecl_file = util_alloc_full_path(ecl_config->root_path , ext_name);
  return ecl_file;
}


void ecl_config_free(ecl_config_type * ecl_config) {
  if (ecl_config->root_path != NULL) free(ecl_config->root_path);
  hash_free(ecl_config->enkf_ecl_kw);
  free(ecl_config);
}

