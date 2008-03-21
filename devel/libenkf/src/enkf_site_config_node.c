#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <util.h>
#include <enkf_site_config_node.h>



struct enkf_site_config_node_struct {
  char                          * key;
  char 			        * value;
  char 			        * default_value;
  int  			          selection_size;
  char 			       ** selection_set;       
  bool                            value_set;
  
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




enkf_site_config_node_type * enkf_site_config_node_alloc(const char * key, const char * default_value , int selection_size , const char ** selection_set) {
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


