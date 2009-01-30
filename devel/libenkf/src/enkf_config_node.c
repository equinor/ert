#include <enkf_types.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stringlist.h>
#include <enkf_macros.h>
#include <enkf_config_node.h> 
#include <util.h>
#include <path_fmt.h>

#define ENKF_CONFIG_NODE_TYPE_ID 776104

struct enkf_config_node_struct {
  int                     __type_id; 
  config_free_ftype     * freef;
  config_activate_ftype * activate;
  enkf_impl_type     	  impl_type;
  enkf_var_type      	  var_type; 
  
  stringlist_type       * obs_keys;         /* Keys of observations which observe this node. */
  char               	* key;
  path_fmt_type         * enkf_infile_fmt;  /* Format used to load in file from forward model - one %d (if present) is replaced with report_step. */
  path_fmt_type	     	* enkf_outfile_fmt; /* Name of file which is written by EnKF, and read by the forward model. */
  void               	* data;             /* This points to the config object of the actual implementation.        */
};



enkf_config_node_type * enkf_config_node_alloc(enkf_var_type              var_type,
					       enkf_impl_type             impl_type,
					       const char               * key , 
					       const char               * enkf_outfile_fmt , 
					       const char               * enkf_infile_fmt  , 
					       const void               * data, 
					       config_free_ftype        * freef,
					       config_activate_ftype    * activate) {
  
  enkf_config_node_type * node = util_malloc( sizeof *node , __func__);
  
  node->data       	= (void *) data;
  node->freef      	= freef;
  node->activate        = activate;  
  node->var_type   	= var_type;
  node->impl_type  	= impl_type;
  node->key        	= util_alloc_string_copy(key);
  node->obs_keys        = stringlist_alloc_new(); 
  node->__type_id       = ENKF_CONFIG_NODE_TYPE_ID;
  if (enkf_infile_fmt != NULL)
    node->enkf_infile_fmt = path_fmt_alloc_path_fmt(enkf_infile_fmt);
  else
    node->enkf_infile_fmt = NULL;

  if (enkf_outfile_fmt != NULL)
    node->enkf_outfile_fmt = path_fmt_alloc_path_fmt(enkf_outfile_fmt);
  else
    node->enkf_outfile_fmt = NULL;
  
  return node;
}


void enkf_config_node_free(enkf_config_node_type * node) {
  if (node->freef   != NULL) node->freef(node->data);
  free(node->key);
  stringlist_free(node->obs_keys);

  if (node->enkf_infile_fmt != NULL) 
    path_fmt_free( node->enkf_infile_fmt );

  if (node->enkf_outfile_fmt != NULL) 
    path_fmt_free( node->enkf_outfile_fmt );
  free(node);
}



/**
   This is the filename used when loading from a completed forward
   model.
*/

char * enkf_config_node_alloc_infile(const enkf_config_node_type * node , int report_step) {
  if (node->enkf_infile_fmt != NULL)
    return path_fmt_alloc_path(node->enkf_infile_fmt , false , report_step);
  else
    return NULL;
}


char * enkf_config_node_alloc_outfile(const enkf_config_node_type * node , int report_step) {
  if (node->enkf_outfile_fmt != NULL)
    return path_fmt_alloc_path(node->enkf_outfile_fmt , false , report_step);
  else
    return NULL;
}



void *  enkf_config_node_get_ref(const enkf_config_node_type * node) { 
  return node->data; 
}



bool enkf_config_node_include_type(const enkf_config_node_type * config_node , int mask) {
  
  enkf_var_type var_type = config_node->var_type;
  if (var_type & mask)
    return true;
  else
    return false;

}


enkf_impl_type enkf_config_node_get_impl_type(const enkf_config_node_type *config_node) { 
  return config_node->impl_type; 
}


enkf_var_type enkf_config_node_get_var_type(const enkf_config_node_type *config_node) { 
  return config_node->var_type; 
}


const char * enkf_config_node_get_key(const enkf_config_node_type * config_node) { return config_node->key; }


const stringlist_type  * enkf_config_node_get_obs_keys(const enkf_config_node_type *config_node) {
  return config_node->obs_keys;
}


void enkf_config_node_add_obs_key(enkf_config_node_type * config_node , const char * obs_key) {
  if (!stringlist_contains(config_node->obs_keys , obs_key))
    stringlist_append_copy(config_node->obs_keys , obs_key);
}

/*****************************************************************/

SAFE_CAST(enkf_config_node , ENKF_CONFIG_NODE_TYPE_ID)
VOID_FREE(enkf_config_node)
