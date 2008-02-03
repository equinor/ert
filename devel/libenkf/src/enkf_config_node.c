#include <enkf_types.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <enkf_macros.h>
#include <enkf_config_node.h> 
#include <util.h>

struct enkf_config_node_struct {
  config_free_ftype  * freef;
  enkf_impl_type       impl_type;
  enkf_var_type        var_type; 
  char 		     * ensfile;          
  char 		     * eclfile;          
  void               * data; /* This points to the config object of the actual implementation. */
} ;



enkf_config_node_type * enkf_config_node_alloc(enkf_var_type              var_type,
					       enkf_impl_type             impl_type,
					       const char               * ensfile , 
					       const char               * eclfile , 
					       const void               * data, 
					       config_free_ftype        * freef) {
  
  enkf_config_node_type * node = malloc( sizeof *node);
  node->data       = (void *) data;
  node->freef      = freef;
  node->var_type   = var_type;
  node->impl_type  = impl_type;
  node->ensfile    = util_alloc_string_copy(ensfile);
  node->eclfile    = util_alloc_string_copy(eclfile);

  return node;
}


void enkf_config_node_free(enkf_config_node_type * node) {
  if (node->freef   != NULL) node->freef(node->data);
  if (node->ensfile != NULL) free(node->ensfile);
  if (node->eclfile != NULL) free(node->eclfile);
  free(node);
}


const void *  enkf_config_node_get_ref(const enkf_config_node_type * node) { 
  return node->data; 
}

bool enkf_config_node_include_type(const enkf_config_node_type * config_node , int mask) {
  enkf_var_type var_type;
  if (config_node == NULL) 
    var_type = ecl_static;
  else
    var_type = config_node->var_type;

  
  if (var_type & mask)
    return true;
  else
    return false;
}


enkf_impl_type enkf_config_node_get_impl_type(const enkf_config_node_type *config_node) { 
  if (config_node == NULL)
    return STATIC;
  else
    return config_node->impl_type; 
}

enkf_var_type enkf_config_node_get_var_type(const enkf_config_node_type *config_node) { 
  if (config_node == NULL)
    return ecl_static;
  else
  return config_node->var_type; 
}


const char * enkf_config_node_get_ensfile_ref(const enkf_config_node_type * config_node) { return config_node->ensfile; }
const char * enkf_config_node_get_eclfile_ref(const enkf_config_node_type * config_node) { return config_node->eclfile; }


VOID_FREE(enkf_config_node)
