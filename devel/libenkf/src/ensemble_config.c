#include <stdlib.h>
#include <string.h>
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
#include <gen_data_config.h>
#include <multflt_config.h>
#include <thread_pool.h>
#include <meas_matrix.h>
#include <enkf_types.h>
#include <analysis.h>
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
#include <pilot_point_config.h>
#include <gen_data_config.h>
#include <stringlist.h>
#include <ensemble_config.h>
#include <config.h>
#include <gen_data_config.h>
#include <pthread.h>                /* Must have rw locking on the config_nodes ... */




struct ensemble_config_struct {
  int  		   ens_size;        /*  The size of the ensemble  */
  hash_type       *config_nodes;    /*  A hash of enkf_config_node instances - which again conatin pointers to e.g. field_config objects.  */
};




static ensemble_config_type * ensemble_config_alloc_empty(int ens_size) {
  if (ens_size <= 0)
    util_exit("%s: ensemble size must be > 0 \n",__func__);
  {
    ensemble_config_type * ensemble_config = util_malloc(sizeof * ensemble_config , __func__);
    ensemble_config->ens_size     = ens_size;
    ensemble_config->config_nodes = hash_alloc();
    
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
    util_abort("%s: ens node:\"%s\" does not exist \n",__func__ , key);
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
    config_free_ftype     * freef = NULL;
    config_activate_ftype * activate = NULL;
    switch(impl_type) {
    case(FIELD):
      freef             = field_config_free__;
      activate          = field_config_activate__;
      break;
    case(MULTZ):
      freef             = multz_config_free__;
      activate          = multz_config_activate__;
      break;
    case(RELPERM):
      freef             = relperm_config_free__;
      break;
    case(WELL):
      freef             = well_config_free__;
      break;
    case(MULTFLT):
      freef             = multflt_config_free__;
      activate          = multflt_config_activate__;
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
      activate          = gen_data_config_activate__;
      break;
    default:
      util_abort("%s : invalid implementation type: %d - aborting \n",__func__ , impl_type);
    }
    
    {
      enkf_config_node_type * node = enkf_config_node_alloc(enkf_type , impl_type , key , enkf_outfile , enkf_infile , data , freef , activate);
      hash_insert_hash_owned_ref(ensemble_config->config_nodes , key , node , enkf_config_node_free__);
    }
  }
}






void ensemble_config_add_well(ensemble_config_type * ensemble_config , const char *well_name , int size, const char ** var_list) {
  ensemble_config_add_node(ensemble_config , well_name , dynamic , WELL , NULL , NULL , well_config_alloc(well_name , size , var_list));
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
     it is difficult/impossible to let the config system handle run
     time validation of the input.
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
    const char * template_file;   

    if (stringlist_get_size(tokens) == 4)
      template_file = stringlist_iget(tokens , 3);
    else
      template_file = NULL;
    
    {
      gen_data_config_type * gen_data_config;

      if (template_file == NULL)
	gen_data_config = gen_data_config_alloc(ASCII , ASCII , init_fmt);
      else
	gen_data_config = gen_data_config_alloc_with_template(ASCII , template_file , "<DATA>" , init_fmt);

      ensemble_config_add_node(ensemble_config , key , parameter , GEN_DATA , ecl_file , NULL , gen_data_config);
    }
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
    const int    size            = stringlist_get_size(tokens);
    const char * key             = stringlist_iget(tokens , 0);
    const char * var_type_string = stringlist_iget(tokens , 1);
    
    if (strcmp(var_type_string , "DYNAMIC") == 0) {
      const char * truncation         = NULL;
      const char ** truncation_values = NULL;
      if (size >= 3) {
	truncation         = stringlist_iget(tokens , 2);
	truncation_values  = stringlist_iget_argv(tokens , 3);
      }
      ensemble_config_add_node(ensemble_config , key , dynamic , FIELD , NULL , NULL , field_config_alloc_dynamic(key , truncation , truncation_values , grid));
    } else if (strcmp(var_type_string , "PARAMETER") == 0) {
      const char *  ecl_file     	  = stringlist_iget(tokens , 2);
      const char *  init_string  	  = stringlist_iget(tokens , 3);
      const char *  output_transform_name = stringlist_iget(tokens , 4);
      const char ** config_files 	  = stringlist_iget_argv(tokens , 5);
      int   num_config_files     	  = stringlist_get_size(tokens) - 5; 
      int   init_mode = -13;
      if (util_sscanf_int(init_string , &init_mode)) 
	ensemble_config_add_node(ensemble_config , key , parameter   , FIELD , ecl_file , NULL , 
				 field_config_alloc_parameter(key , ecl_file , output_transform_name , grid ,init_mode , num_config_files , config_files));
      else 
	util_abort("%s: init_mode:%s must be a valid integer - aborting \n",__func__ , init_string);
    } else if (strcmp(var_type_string , "GENERAL") == 0) {
      const char * enkf_outfile = stringlist_iget(tokens , 2); /* Out before in ?? */
      const char * enkf_infile  = stringlist_iget(tokens , 3);
      const char * init_fmt     = stringlist_iget(tokens , 4);
      ensemble_config_add_node(ensemble_config , key , dynamic , FIELD , enkf_outfile , enkf_infile , field_config_alloc_general(key , enkf_outfile , grid , ecl_float_type , init_fmt));
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

    ensemble_config_add_node(ensemble_config , key , dynamic , SUMMARY , NULL , NULL , summary_config_alloc(num_variables , variables));
  }
  

  /* GEN_DATA */
  for (i=0; i < config_get_occurences(config , "GEN_DATA"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "GEN_DATA" , i);
    const char * key         = stringlist_iget(tokens , 0);
    const char * config_file = stringlist_iget(tokens , 1);
    
    ensemble_config_add_node(ensemble_config , key , dynamic , GEN_DATA , NULL , NULL , gen_data_config_fscanf_alloc(config_file));
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

    
  return ensemble_config;
}




char ** ensemble_config_alloc_keylist(const ensemble_config_type * config , int *keys) {
  *keys = hash_get_size(config->config_nodes);
  return hash_alloc_keylist(config->config_nodes);
}




