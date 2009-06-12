#include <enkf_types.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stringlist.h>
#include <enkf_macros.h>
#include <enkf_config_node.h> 
#include <util.h>
#include <path_fmt.h>
#include <bool_vector.h>
#include <field_config.h>
#include <multflt_config.h>
#include <gen_data_config.h>
#include <gen_kw_config.h>
#include <summary_config.h>
#include <havana_fault_config.h>

#define ENKF_CONFIG_NODE_TYPE_ID 776104

struct enkf_config_node_struct {
  int                     __type_id; 
  enkf_impl_type     	  impl_type;
  enkf_var_type      	  var_type; 

  bool_vector_type      * internalize;      /* Should this node be internalized - observe that question of what to internalize is MOSTLY handled at a hight level - without consulting this variable. Can be NULL. */ 
  stringlist_type       * obs_keys;         /* Keys of observations which observe this node. */
  char               	* key;
  path_fmt_type         * enkf_infile_fmt;  /* Format used to load in file from forward model - one %d (if present) is replaced with report_step. */
  path_fmt_type	     	* enkf_outfile_fmt; /* Name of file which is written by EnKF, and read by the forward model. */
  void               	* data;             /* This points to the config object of the actual implementation.        */

  /*****************************************************************/
  /* Function pointers to methods working on the underlying config object. */
  get_data_size_ftype   * get_data_size;    /* Function pointer to ask the underlying config object of the size - i.e. number of elements. */
  config_free_ftype     * freef;
};



enkf_config_node_type * enkf_config_node_alloc(enkf_var_type              var_type,
					       enkf_impl_type             impl_type,
					       const char               * key , 
					       const char               * enkf_outfile_fmt , 
					       const char               * enkf_infile_fmt  , 
					       const void               * data) {

  
  enkf_config_node_type * node = util_malloc( sizeof *node , __func__);

  node->internalize     = NULL;
  node->data       	= (void *) data;
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
  

  /* Some manual inheritance: */
  node->get_data_size = NULL;
  node->freef         = NULL; 

  {  
    switch(impl_type) {
    case(FIELD):
      node->freef             = field_config_free__;
      node->get_data_size     = field_config_get_data_size__;  
      break;
    case(STATIC):
      break;
    case(GEN_KW):
      node->freef             = gen_kw_config_free__;
      node->get_data_size     = gen_kw_config_get_data_size__;
      break;
    case(SUMMARY):
      node->freef             = summary_config_free__;
      node->get_data_size     = summary_config_get_data_size__;
      break;
    case(MULTFLT):
      node->freef             = multflt_config_free__;
      node->get_data_size     = multflt_config_get_data_size__;
      break;
    case(HAVANA_FAULT):
      node->freef             = havana_fault_config_free__;
      node->get_data_size     = havana_fault_config_get_data_size__;
      break;
    case(GEN_DATA):
      node->freef             = gen_data_config_free__;
      node->get_data_size     = gen_data_config_get_data_size__;
      break;
    default:
      util_abort("%s : invalid implementation type: %d - aborting \n",__func__ , impl_type);
    }
  }
  return node;
}


/**
   Invokes the get_data_size() function of the underlying node object.
*/

int enkf_config_node_get_data_size( const enkf_config_node_type * node) {
  return node->get_data_size( node->data );
}

void enkf_config_node_free(enkf_config_node_type * node) {
  /* Freeing the underlying node object. */
  if (node->freef   != NULL) node->freef(node->data);
  free(node->key);
  stringlist_free(node->obs_keys);

  if (node->enkf_infile_fmt != NULL) 
    path_fmt_free( node->enkf_infile_fmt );

  if (node->enkf_outfile_fmt != NULL) 
    path_fmt_free( node->enkf_outfile_fmt );

  if (node->internalize != NULL)
    bool_vector_free( node->internalize );

  free(node);
}


void enkf_config_node_set_internalize(enkf_config_node_type * node, int report_step) {
  if (node->internalize == NULL)
    node->internalize = bool_vector_alloc(report_step , false);
  bool_vector_iset( node->internalize , report_step , true);
}


void enkf_config_node_init_internalization(enkf_config_node_type * node) {
  if (node->internalize != NULL)
    bool_vector_reset( node->internalize );
}

/* Query function: */
bool enkf_config_node_internalize(const enkf_config_node_type * node, int report_step) {
  if (node->internalize == NULL)
    return false;
  else
    return bool_vector_safe_iget( node->internalize , report_step); /* Will return default value if report_step is beyond size. */
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
