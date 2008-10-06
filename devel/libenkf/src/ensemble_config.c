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
#include <ecl_grid.h>
#include <time.h>
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
  bool            *keep_runpath;    /* Whether the runpath should be kept for the various members. */
};




static ensemble_config_type * ensemble_config_alloc_empty(int ens_size) {
  if (ens_size <= 0)
    util_exit("%s: ensemble size must be > 0 \n",__func__);
  {
    ensemble_config_type * ensemble_config = util_malloc(sizeof * ensemble_config , __func__);
    ensemble_config->ens_size     = ens_size;
    ensemble_config->config_nodes = hash_alloc();
    ensemble_config->keep_runpath = util_malloc(ens_size * sizeof * ensemble_config->keep_runpath , __func__);
    for (int i = 0; i < ens_size; i++)
      ensemble_config->keep_runpath[i] = false;
    
    return ensemble_config;
  }
}


enkf_impl_type ensemble_config_impl_type(const ensemble_config_type *ensemble_config, const char * ecl_kw_name) {
  enkf_impl_type impl_type = INVALID;

  if (hash_has_key(ensemble_config->config_nodes , ecl_kw_name)) {
    enkf_config_node_type * node = hash_get(ensemble_config->config_nodes , ecl_kw_name);
    impl_type = enkf_config_node_get_impl_type(node);
  } else
    util_abort("%s: internal error: asked for implementation type of unknown node:%s \n",__func__ , ecl_kw_name);

  return impl_type;
}


enkf_var_type ensemble_config_var_type(const ensemble_config_type *ensemble_config, const char * ecl_kw_name) {
  enkf_var_type var_type = invalid;

  if (hash_has_key(ensemble_config->config_nodes , ecl_kw_name)) {
    enkf_config_node_type * node = hash_get(ensemble_config->config_nodes , ecl_kw_name);
    var_type = enkf_config_node_get_var_type(node);
  } else
    util_abort("%s: internal error: asked for implementation type of unknown node:%s \n",__func__ , ecl_kw_name);

  return var_type;
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


void ensemble_config_add_well(ensemble_config_type * ensemble_config , const char *well_name , int size, const char ** var_list) {
  ensemble_config_add_node(ensemble_config , well_name , ecl_summary , WELL , NULL , NULL , well_config_alloc(well_name , size , var_list));
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
  
  item = config_add_item(config , "GEN_DATA" , false , true);
  config_item_set_argc_minmax(item , 2 , 2 ,  (const config_item_types [4]) { CONFIG_STRING , CONFIG_EXISTING_FILE});

  item = config_add_item(config , "HAVANA_FAULT" , false , true);
  config_item_set_argc_minmax(item , 2 , 2 ,  (const config_item_types [4]) { CONFIG_STRING , CONFIG_EXISTING_FILE});


  item = config_add_item(config , "WELL" , false , true);
  config_item_set_argc_minmax(item , 2 , -1 ,  NULL);
  
  item = config_add_item(config , "SUMMARY" , false , true);
  config_item_set_argc_minmax(item , 2 , -1 ,  NULL);
  

  /* 
     The way config info is entered for fields is unfortunate because
     it is difficult/impossible to let the config system handle run time
     validation of the input.
  */
     
  item = config_add_item(config , "FIELD" , false , true);
  config_item_set_argc_minmax(item , 2 , -1 ,  NULL);
}




ensemble_config_type * ensemble_config_alloc(const config_type * config , const ecl_grid_type * grid) {
  int i;
  ensemble_config_type * ensemble_config = ensemble_config_alloc_empty( strtol(config_get(config , "NUM_REALIZATIONS") , NULL , 10) );

  /* MULTZ */
  for (i=0; i < config_get_occurences(config , "MULTZ"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "MULTZ" , i);
    const char * key         = stringlist_iget(tokens , 0);
    const char * ecl_file    = stringlist_iget(tokens , 1);
    const char * config_file = stringlist_iget(tokens , 2);
    int   nx,ny,nz;
    
    ecl_grid_get_dims(grid , &nx , &ny , &nz , NULL);
    ensemble_config_add_node(ensemble_config , key , parameter , MULTZ , ecl_file , NULL , multz_config_fscanf_alloc(config_file , nx , ny , nz));
  }


  /* MULTFLT */
  for (i=0; i < config_get_occurences(config , "MULTFLT"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "MULTFLT" , i);
    const char * key         = stringlist_iget(tokens , 0);
    const char * ecl_file    = stringlist_iget(tokens , 1);
    const char * config_file = stringlist_iget(tokens , 2);

    ensemble_config_add_node(ensemble_config , key , parameter , MULTFLT , ecl_file , NULL , multflt_config_fscanf_alloc(config_file));
  }


  /* GEN_PARAM */
  for (i=0; i < config_get_occurences(config , "GEN_PARAM"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "GEN_PARAM" , i);
    const char * key         = stringlist_iget(tokens , 0);
    const char * ecl_file    = stringlist_iget(tokens , 1);
    const char * init_fmt    = stringlist_iget(tokens , 2);
    const char * ecl_template;   

    if (stringlist_get_size(tokens) == 4)
      ecl_template = stringlist_iget(tokens , 3);
    else
      ecl_template = NULL;
    
    ensemble_config_add_node(ensemble_config , key , parameter , GEN_PARAM , ecl_file , NULL , gen_param_config_alloc( init_fmt , ecl_template ));
  }
  

  /* HAVANA_FAULT */
  for (i=0; i < config_get_occurences(config , "HAVANA_FAULT"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "HAVANA_FAULT" , i);
    const char * key         = stringlist_iget(tokens , 0);
    const char * config_file = stringlist_iget(tokens , 1);
    
    ensemble_config_add_node(ensemble_config , key , parameter , HAVANA_FAULT , NULL , NULL , havana_fault_config_fscanf_alloc(config_file));
  }


  /* EQUIL */
  for (i=0; i < config_get_occurences(config , "EQUIL"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "EQUIL" , i);
    const char * key         = stringlist_iget(tokens , 0);
    const char * ecl_file    = stringlist_iget(tokens , 1);
    const char * config_file = stringlist_iget(tokens , 2);

    ensemble_config_add_node(ensemble_config , key , parameter , EQUIL , ecl_file , NULL , multflt_config_fscanf_alloc(config_file));
  }


  /* FIELD */
  for (i=0; i < config_get_occurences(config , "FIELD"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "FIELD" , i);
    const char * key             = stringlist_iget(tokens , 0);
    const char * var_type_string = stringlist_iget(tokens , 1);
    
    if (strcmp(var_type_string , "DYNAMIC") == 0)
      ensemble_config_add_node(ensemble_config , key , ecl_restart , FIELD , NULL , NULL , field_config_alloc_dynamic(key , grid));
    else if (strcmp(var_type_string , "PARAMETER") == 0) {
      const char *  ecl_file     = stringlist_iget(tokens , 2);
      const char *  init_string  = stringlist_iget(tokens , 3);
      const char ** config_files = stringlist_iget_argv(tokens , 4);
      int   num_config_files     = stringlist_get_size(tokens) - 4; 
      int init_mode = -13;
      if (util_sscanf_int(init_string , &init_mode)) 
	ensemble_config_add_node(ensemble_config , key , parameter   , FIELD , ecl_file , NULL , field_config_alloc_parameter(key , grid , 0 , init_mode , num_config_files , config_files));
      else 
	util_abort("%s: init_mode:%s must be a valid integer - aborting \n",__func__ , init_string);
    } else if (strcmp(var_type_string , "GENERAL") == 0) {
      const char * enkf_outfile = stringlist_iget(tokens , 2);
      const char * enkf_infile  = stringlist_iget(tokens , 3);
      const char * init_fmt     = stringlist_iget(tokens , 4);
      ensemble_config_add_node(ensemble_config , key , ecl_restart , FIELD , enkf_outfile , enkf_infile , field_config_alloc_general(key , grid , init_fmt));
    } else 
      util_abort("%s: FIELD type: %s is not recognized\n",__func__ , var_type_string);
  }


  /* WELL */
  for (i=0; i < config_get_occurences(config , "WELL"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "WELL" , i);
    const char *  well_name   = stringlist_iget(tokens , 0);
    const char ** variables   = stringlist_iget_argv(tokens , 1);
    int   num_variables       = stringlist_get_size(tokens) - 1; 

    ensemble_config_add_well(ensemble_config , well_name , num_variables, variables);
  }


  /* SUMMARY */
  for (i=0; i < config_get_occurences(config , "SUMMARY"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "SUMMARY" , i);
    const char * key        = stringlist_iget(tokens , 0);
    const char ** variables = stringlist_iget_argv(tokens , 1);
    int   num_variables     = stringlist_get_size(tokens) - 1; 

    ensemble_config_add_node(ensemble_config , key , ecl_summary , SUMMARY , NULL , NULL , summary_config_alloc(num_variables , variables));
  }
  

  /* GEN_DATA */
  for (i=0; i < config_get_occurences(config , "GEN_DATA"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "GEN_DATA" , i);
    const char * key         = stringlist_iget(tokens , 0);
    const char * config_file = stringlist_iget(tokens , 1);
    
    ensemble_config_add_node(ensemble_config , key , parameter , GEN_DATA , NULL , NULL , gen_data_config_fscanf_alloc(config_file));
  }


  /* GEN_KW */
  for (i=0; i < config_get_occurences(config , "GEN_KW"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "GEN_KW" , i);
    const char * key           = stringlist_iget(tokens , 0);
    const char * template_file = stringlist_iget(tokens , 1);
    const char * target_file   = stringlist_iget(tokens , 2);
    const char * config_file   = stringlist_iget(tokens , 3);
    
    ensemble_config_add_node(ensemble_config , key , parameter , GEN_KW , target_file , NULL , gen_kw_config_fscanf_alloc(config_file , template_file));
  }


  /*****************************************************************/
  /* KEEP_RUNPATH - this parser is *EXTREMELY* primitive */
  for (i=0; i < config_get_occurences(config , "KEEP_RUNPATH"); i++) {
    const stringlist_type * _tokens = config_iget_stringlist_ref(config , "KEEP_RUNPATH" , i);
    const char ** token_list = stringlist_get_argv(_tokens);
    int           tokens     = stringlist_get_size(_tokens);
    
    int token_index   = 0;
    int prev_iens     = -1;
    bool range_active = false;
    do {
      if (token_list[token_index][0] == ',')
	token_index++;
      else {
	if (token_list[token_index][0] == '-') {
	  if (prev_iens == -1) 
	    util_abort("%s: something rotten - lonesome dash \n",__func__);

	  range_active = true;
	} else {
	  int iens,iens1,iens2;
	  if (util_sscanf_int(token_list[token_index] , &iens2)) {
	    if (range_active)
	      iens1 = prev_iens;
	    else {
	      iens1     = iens2;
	      prev_iens = iens2;
	    }
	    for (iens = iens1; iens <= iens2; iens++)
	      if (iens2 < ensemble_config->ens_size) 
		ensemble_config->keep_runpath[iens] = true;
	    
	    range_active = false;  
	  } else 
	    util_abort("%s: something wrong when parsing: \"%s\" to integer \n",__func__ , token_list[token_index]);
	}
	token_index++;
      }
    } while (token_index < tokens);
  }
  /*****************************************************************/

    
  return ensemble_config;
}


bool ensemble_config_iget_keep_runpath(const ensemble_config_type * config , int iens) {
  if (iens >= 0 && iens < config->ens_size)
    return config->keep_runpath[iens];
  else {
    util_abort("%s ... \n",__func__);
    return false; /* Compiler shut up */
  }
}


char ** ensemble_config_alloc_keylist(const ensemble_config_type * config , int *keys) {
  *keys = hash_get_size(config->config_nodes);
  return hash_alloc_keylist(config->config_nodes);
}




