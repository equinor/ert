#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <util.h>
#include <hash.h>
#include <enkf_site_config.h>

typedef bool   (validate_ftype)                   (const enkf_site_config_type * , const enkf_site_config_node_type *);


struct enkf_site_config_struct {
  hash_type * config;
};



struct enkf_site_config_node_struct {
  char                          * key;
  char 			        * value;
  char 			        * default_value;
  int  			          selection_size;
  char 			       ** selection_set;       
  bool                            value_set;
  validate_ftype                * validate;
};


void enkf_config_node_set_value(enkf_site_config_node_type * node , const char * value) {
  if (node->value_set)
    free(node->value);

  if (node->selection_size > 0) {
    bool value_found = false;
    int is;
    for (is=0; is < node->selection_size; is++) 
      if (strcmp(value , node->selection_set[is]) == 0) {
	value_found = true;
	break;
      }
    if (!value_found) {
      fprintf(stderr,"%s: for site_config_key:%s the value:%s is not among the allowed values - aborting \n",__func__ , node->key , value);
      fprintf(stderr,"%s: allowed values: ",__func__);
      for (is=0; is < node->selection_size; is++) 
	fprintf(stderr," %s ",node->selection_set[is]);
      fprintf(stderr,"\n");
      abort();
    }
  }
  node->value     = util_alloc_string_copy(value);
  node->value_set = true;
}



enkf_site_config_node_type * enkf_site_config_node_alloc(const char * key, const char * default_value , int selection_size , const char ** selection_set , validate_ftype * validate) {
  enkf_site_config_node_type * node = util_malloc(sizeof * node , __func__);
  node->key = util_alloc_string_copy(key);
  node->default_value  = util_alloc_string_copy(default_value);
  node->selection_size = selection_size;
  node->value          = NULL;
  if (selection_size > 0) 
    node->selection_set = util_alloc_stringlist_copy(selection_set , selection_size);
  else
    node->selection_set = NULL;
  node->value_set       = false;
  if (default_value != NULL)
    enkf_config_node_set_value(node , default_value);
  node->validate = validate;
  return node;
}




void enkf_site_config_node_free(enkf_site_config_node_type * node) {
  free(node->key);
  if (node->value != NULL)         free(node->value);
  if (node->default_value != NULL) free(node->default_value);
  if (node->selection_size > 0) {
    int is;
    for (is=0; is < node->selection_size; is++) 
      free(node->selection_set[is]);
    free(node->selection_set);
  }
  free(node);
}


void enkf_site_config_node_free__(void * node) {
  enkf_site_config_node_free( (enkf_site_config_node_type *) node ); 
}


bool enkf_site_config_node_validate(const enkf_site_config_type * config , const enkf_site_config_node_type * node) {
  if (node->validate != NULL)
    return node->validate(config , node);
  else
    return true;
}


static bool enkf_site_config_validate_queue(const enkf_site_config_type * config , const enkf_site_config_node_type * node) {
  return true;
}
/*****************************************************************/


void enkf_site_config_add_node(enkf_site_config_type * site , const char * key , const char * default_value , int selection_size , const char ** selection_set , validate_ftype * validate) {
  hash_insert_hash_owned_ref(site->config , key , enkf_site_config_node_alloc(key , default_value , selection_size , selection_set , validate) , enkf_site_config_node_free__);
}


enkf_site_config_type * enkf_site_config_bootstrap(const char * _config_file) {
  const char * config_file = getenv("ENKF_SITE_CONFIG");
  if (config_file == NULL)
    config_file = _config_file;
  
  if (config_file == NULL) {
    fprintf(stderr,"%s: main enkf_config file is not set. Use environment variable \"ENKF_SITE_CONFIG\" - or recompile - aborting \n",__func__);
    abort();
  }
  if (util_file_exists(config_file)) {
    enkf_site_config_type * site;
    site = util_malloc(sizeof * site , __func__);
    enkf_site_config_add_node(site , "QUEUE_SYSTEM" , NULL , 2 , (const char *[2]) {"LSF" , "LOCAL"} , enkf_site_config_validate_queue);
    return site;
  } else {
    fprintf(stderr,"%s: main config_file: %s not found - aborting \n",__func__ , config_file);
    abort();
  }
}



