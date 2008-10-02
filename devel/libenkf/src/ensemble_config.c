#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <util.h>
#include <hash.h>
#include <set.h>
#include <multz_config.h>
#include <enkf_config_node.h>
#include <path_fmt.h>
#include <enkf_types.h>
#include <well_config.h>
#include <field_config.h>
#include <equil_config.h>
#include <gen_param_config.h>
#include <multflt_config.h>
#include <well_obs.h>
#include <thread_pool.h>
#include <obs_node.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <enkf_types.h>
#include <analysis.h>
#include <enkf_obs.h>
#include <sched_file.h>
#include <enkf_fs.h>
#include <void_arg.h>
#include <gen_kw_config.h>
#include <enkf_config.h>
#include <ecl_grid.h>
#include <time.h>
#include <enkf_site_config.h>
#include <job_queue.h>
#include <lsf_driver.h>
#include <local_driver.h>
#include <rsh_driver.h>
#include <relperm_config.h>
#include <summary.h>
#include <summary_config.h>
#include <havana_fault_config.h>
#include <ext_joblist.h>
#include <gen_data.h>
#include <gen_data_config.h>
#include <stringlist.h>
#include <ensemble_config.h>
#include <config.h>


struct ensemble_config_struct {
  int  		   ens_size;        /* The size of the ensemble */
  hash_type       *config_nodes;    /* A hash of enkf_config_node instances - which again conatin pointers to e.g. field_config objects. */
};




ensemble_config_type * ensemble_config_alloc_empty(int ens_size) {
  if (ens_size <= 0)
    util_exit("%s: ensemble size must be > 0 \n",__func__);
  {
    ensemble_config_type * ensemble_config = util_malloc(sizeof * ensemble_config , __func__);
    ensemble_config->ens_size     = ens_size;
    ensemble_config->config_nodes = hash_alloc();
    return ensemble_config;
  }
}

ensemble_config_type * ensemble_config_alloc(const config_type * config) {
  ensemble_config_type * ensemble_config = ensemble_config_alloc_empty( strtol(config_get(config , "NUM_REALIZATIONS") , NULL , 10) );
  
  return ensemble_config;
}



void ensemble_config_free(ensemble_config_type * ensemble_config) {
  hash_free(ensemble_config->config_nodes);
  free(ensemble_config);
}


int ensemble_config_get_size(const ensemble_config_type * ensemble_config) { 
  return ensemble_config->ens_size;
}



bool ensemble_config_has_key(const ensemble_config_type * ensemble_config , const char * key) {
  return hash_has_key( ensemble_config->config_nodes , key);
}



const enkf_config_node_type * ensemble_config_get_node(const ensemble_config_type * ensemble_config, const char * key) {
  if (hash_has_key(ensemble_config->config_nodes , key)) {
    enkf_config_node_type * node = hash_get(ensemble_config->config_nodes , key);
    return node;
  } else {
    util_abort("%s: ens node:%s does not exist \n",__func__ , key);
    return NULL; /* Compiler shut up */
  }
}



void ensemble_config_add_node(ensemble_config_type * ensemble_config , 
			      const char    * key      	   , 
			      enkf_var_type enkf_type  	   , 
			      enkf_impl_type impl_type 	   ,    
			      const char   * enkf_outfile  , /* Written by EnKF and read by forward model */
			      const char   * enkf_infile   , /* Written by forward model and read by EnKF */ 
			      const void   * data) {

  if (ensemble_config_has_key(ensemble_config , key)) 
    util_abort("%s: a configuration object:%s has already been added - aborting \n",__func__ , key);
  
  {
    config_free_ftype * freef = NULL;
    switch(impl_type) {
    case(FIELD):
      freef             = field_config_free__;
      break;
    case(MULTZ):
      freef             = multz_config_free__;
      break;
    case(RELPERM):
      freef             = relperm_config_free__;
      break;
    case(WELL):
      freef             = well_config_free__;
      break;
    case(MULTFLT):
      freef             = multflt_config_free__;
      break;
    case(EQUIL):
      freef             = equil_config_free__;
      break;
    case(STATIC):
      freef             = NULL; 
      break;
    case(GEN_KW):
      freef             = gen_kw_config_free__;
      break;
    case(SUMMARY):
      freef             = summary_config_free__;
      break;
    case(HAVANA_FAULT):
      freef             = havana_fault_config_free__;
      break;
    case(GEN_DATA):
      freef             = gen_data_config_free__;
      break;
    case(GEN_PARAM):
      freef             = gen_param_config_free__;
      break;
    default:
      util_abort("%s : invalid implementation type: %d - aborting \n",__func__ , impl_type);
    }
    
    {
      enkf_config_node_type * node = enkf_config_node_alloc(enkf_type , impl_type , key , enkf_outfile , enkf_infile , data , freef);
      hash_insert_hash_owned_ref(ensemble_config->config_nodes , key , node , enkf_config_node_free__);
    }
  }
}



void ensemble_config_add_config_items(config_type * config) {
  config_item_type * item;

  item = config_add_item(config , "MULTZ" , false , true);
  config_item_set_argc_minmax(item , 3 , 3 ,  (const config_item_types [3]) { CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});
  
  item = config_add_item(config , "MULTFLT" , false , true);
  config_item_set_argc_minmax(item , 3 , 3 ,  (const config_item_types [3]) { CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});
  
  item = config_add_item(config , "EQUIL" , false , true);
  config_item_set_argc_minmax(item , 3 , 3 ,  (const config_item_types [3]) { CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});
  
  item = config_add_item(config , "GEN_KW" , false , true);
  config_item_set_argc_minmax(item , 4 , 4 ,  (const config_item_types [4]) { CONFIG_STRING , CONFIG_EXISTING_FILE , CONFIG_STRING , CONFIG_EXISTING_FILE});
  
  item = config_add_item(config , "GEN_PARAM" , false , true);
  config_item_set_argc_minmax(item , 3 , 4 ,  (const config_item_types [4]) { CONFIG_STRING , CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});
  
  item = config_add_item(config , "WELL" , false , true);
  config_item_set_argc_minmax(item , 2 , -1 ,  NULL);
  
  item = config_add_item(config , "SUMMARY" , false , true);
  config_item_set_argc_minmax(item , 2 , -1 ,  NULL);
  
  item = config_add_item(config , "FIELD" , false , true);
  config_item_set_argc_minmax(item , 2 , -1 ,  NULL);
}
