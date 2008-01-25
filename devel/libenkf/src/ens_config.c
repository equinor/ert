#include <stdlib.h>
#include <stdio.h>
#include <util.h>
#include <ens_config.h>

struct ens_config_struct {
  int  		    size;
  char 		   *root_path;
  char             *ext_path;
  char             *_ens_path;
};


/*****************************************************************/

static void ens_config_set_path2(ens_config_type * ens_config , const char *root_path , const char *ext_path) {

  if (root_path != 0)
    ens_config->root_path = util_realloc_string_copy(ens_config->root_path , root_path);
  
  if (ext_path != 0) {
    if (ext_path[0] == '/') {
      fprintf(stderr,"%s: ext_path must be relative - i.e. can not start with \'/\' - aborting \n",__func__);
      abort();
    }
    ens_config->ext_path = util_realloc_string_copy(ens_config->ext_path , ext_path);
  }
  
  
  if (ens_config->ext_path != NULL && ens_config->root_path != NULL) {
    ens_config->_ens_path = util_realloc_full_path(ens_config->_ens_path , ens_config->root_path , ens_config->ext_path);
    util_make_path(ens_config->_ens_path);
  }
}


void ens_config_set_root_path(ens_config_type * ens_config , const char * root_path) {
  ens_config_set_path2(ens_config , root_path , NULL);
}

void ens_config_set_ext_path(ens_config_type * ens_config , const char * ext_path) {
  ens_config_set_path2(ens_config , NULL , ext_path);
}


ens_config_type * ens_config_alloc(int size , const char * root_path) {
  ens_config_type * ens_config = malloc(sizeof *ens_config);
  ens_config->size      = size;
  ens_config->root_path = NULL;
  ens_config->ext_path  = NULL;
  ens_config->_ens_path = NULL;

  ens_config_set_root_path(ens_config , root_path);
  return ens_config;
}


char * ens_config_alloc_ensname(const ens_config_type * ens_config, const char * ext_name) {
  if (ens_config->_ens_path == NULL) {
    fprintf(stderr,"%s: must set a valid ext_path with ens_config_set_ext_path() first - aborting \n",__func__);
    abort();
  } 
  {
    char * ens_file = util_alloc_full_path(ens_config->_ens_path , ext_name);
    return ens_file;
  }
}


void ens_config_free(ens_config_type * ens_config) {
  if (ens_config->root_path != NULL) free(ens_config->root_path);
  if (ens_config->ext_path != NULL)  free(ens_config->ext_path);
  if (ens_config->_ens_path != NULL) free(ens_config->_ens_path);
  free(ens_config);
}

