#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <util.h>
#include <hash.h>
#include <set.h>
#include <enkf_config_node.h>
#include <path_fmt.h>
#include <enkf_types.h>
#include <field_config.h>
#include <gen_data_config.h>
#include <thread_pool.h>
#include <meas_matrix.h>
#include <enkf_types.h>
#include <sched_file.h>
#include <enkf_fs.h>
#include <gen_kw_config.h>
#include <ecl_grid.h>
#include <time.h>
#include <job_queue.h>
#include <lsf_driver.h>
#include <local_driver.h>
#include <rsh_driver.h>
#include <summary.h>
#include <summary_config.h>
#include <ext_joblist.h>
#include <gen_data.h>
#include <pilot_point_config.h>
#include <gen_data_config.h>
#include <stringlist.h>
#include <ensemble_config.h>
#include <config.h>
#include <gen_data_config.h>
#include <multflt_config.h>
#include <havana_fault_config.h>
#include <pthread.h>                /* Must have rw locking on the config_nodes ... */
#include <field_trans.h>




struct ensemble_config_struct {
  int  		  	   ens_size;          /*  The size of the ensemble  */
  hash_type       	 * config_nodes;      /*  A hash of enkf_config_node instances - which again conatin pointers to e.g. field_config objects.  */
  field_trans_table_type * field_trans_table; /* A table of the transformations which are available to apply on fields. */
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
  enkf_var_type var_type = INVALID_VAR;

  if (hash_has_key(ensemble_config->config_nodes , ecl_kw_name)) {
    enkf_config_node_type * node = hash_get(ensemble_config->config_nodes , ecl_kw_name);
    var_type = enkf_config_node_get_var_type(node);
  } else
    util_abort("%s: internal error: asked for implementation type of unknown node:%s \n",__func__ , ecl_kw_name);

  return var_type;
}



void ensemble_config_free(ensemble_config_type * ensemble_config) {
  hash_free(ensemble_config->config_nodes);
  field_trans_table_free( ensemble_config->field_trans_table );
  free(ensemble_config);
}


int ensemble_config_get_size(const ensemble_config_type * ensemble_config) { 
  return ensemble_config->ens_size;
}



bool ensemble_config_has_key(const ensemble_config_type * ensemble_config , const char * key) {
  return hash_has_key( ensemble_config->config_nodes , key);
}



enkf_config_node_type * ensemble_config_get_node(const ensemble_config_type * ensemble_config, const char * key) {
  if (hash_has_key(ensemble_config->config_nodes , key)) {
    enkf_config_node_type * node = hash_get(ensemble_config->config_nodes , key);
    return node;
  } else {
    util_abort("%s: ens node:\"%s\" does not exist \n",__func__ , key);
    return NULL; /* Compiler shut up */
  }
}


/** 
    This will remove the config node indexed by key, it will use the
    function hash_safe_del(), which is thread_safe, and will NOT fail
    if the node has already been removed from the hash. 

    However - it is extremely important to ensure that all storage
    nodes (which point to the config nodes) have been deleted before
    calling this function. That is only assured by using
    enkf_main_del_node().
*/


void ensemble_config_del_node(ensemble_config_type * ensemble_config, const char * key) {
  hash_safe_del(ensemble_config->config_nodes , key);
}


enkf_config_node_type *  ensemble_config_add_node(ensemble_config_type * ensemble_config , 
			      const char    * key      	   , 
			      enkf_var_type enkf_type  	   , 
			      enkf_impl_type impl_type 	   ,
			      const char   * enkf_outfile  , /* Written by EnKF and read by forward model */
			      const char   * enkf_infile   , /* Written by forward model and read by EnKF */ 
			      const void   * data) {

  if (ensemble_config_has_key(ensemble_config , key)) 
    util_abort("%s: a configuration object:%s has already been added - aborting \n",__func__ , key);
  
  {
    enkf_config_node_type * node = enkf_config_node_alloc(enkf_type , impl_type , key , enkf_outfile , enkf_infile , data );
    hash_insert_hash_owned_ref(ensemble_config->config_nodes , key , node , enkf_config_node_free__);
    return node;
  }
}






/**
   This function ensures that object contains a node with 'key' and
   type == SUMMARY.
*/
void ensemble_config_ensure_summary(ensemble_config_type * ensemble_config , const char * key) {
  if (hash_has_key(ensemble_config->config_nodes, key)) {
    if (ensemble_config_impl_type(ensemble_config , key) != SUMMARY)
      util_abort("%s: ensemble key:%s already existst - but it is not of summary type\n",__func__ , key);
  } else 
    ensemble_config_add_node(ensemble_config , key , DYNAMIC_RESULT , SUMMARY , NULL , NULL , summary_config_alloc(key));
}



void ensemble_config_add_obs_key(ensemble_config_type * ensemble_config , const char * key, const char * obs_key) {
  enkf_config_node_type * config_node = hash_get(ensemble_config->config_nodes , key);
  enkf_config_node_add_obs_key(config_node , obs_key);
}



void ensemble_config_add_config_items(config_type * config) {
  config_item_type * item;

  item = config_add_item(config , "MULTFLT" , false , true);
  config_item_set_argc_minmax(item , 3 , 3 ,  (const config_item_types [3]) { CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});
  
  item = config_add_item(config , "GEN_KW" , false , true);
  config_item_set_argc_minmax(item , 4 , 5 ,  (const config_item_types [5]) { CONFIG_STRING , CONFIG_EXISTING_FILE , CONFIG_STRING , CONFIG_EXISTING_FILE , CONFIG_EXISTING_FILE});
  
  item = config_add_item(config , "GEN_PARAM" , false , true);
  config_item_set_argc_minmax(item , 5 , 7 ,  NULL);
  
  item = config_add_item(config , "GEN_DATA" , false , true);
  config_item_set_argc_minmax(item , 1 , -1 ,  (const config_item_types [4]) { CONFIG_STRING , CONFIG_EXISTING_FILE});

  item = config_add_item(config , "HAVANA_FAULT" , false , true);
  config_item_set_argc_minmax(item , 2 , 2 ,  (const config_item_types [4]) { CONFIG_STRING , CONFIG_EXISTING_FILE});


  item = config_add_item(config , "SUMMARY" , false , true);
  config_item_set_argc_minmax(item , 1 , 1 ,  NULL);
  

  /* 
     The way config info is entered for fields is unfortunate because
     it is difficult/impossible to let the config system handle run
     time validation of the input.
  */
     
  item = config_add_item(config , "FIELD" , false , true);
  config_item_set_argc_minmax(item , 2 , -1 ,  NULL);
  config_item_add_required_children(item , "GRID");   /* If you are using a FIELD - you must have a grid. */
}


ensemble_config_type * ensemble_config_alloc(const config_type * config , const ecl_grid_type * grid) {
  int i;
  ensemble_config_type * ensemble_config = ensemble_config_alloc_empty( config_iget_as_int(config , "NUM_REALIZATIONS" , 0 , 0));
  ensemble_config->field_trans_table     = field_trans_table_alloc();

  {
    /* MULTFLT depreceation warning added 17/03/09 (svn 1811). */
    if (config_get_occurences(config , "MULTFLT") > 0) {
      printf("*****************************************************************\n");
      printf("**                    W A R N I N G                            **\n");
      printf("** ----------------------------------------------------------  **\n");
      printf("** You have used the keyword \'MULTFLT\' for estimating fault    **\n");
      printf("** transmissibility multipliers. This will be disabled in the  **\n");
      printf("** future. Instead you should use the GEN_KW keyword, look at  **\n");
      printf("** GEN_KW documentation for an example of how to use GEN_KW    **\n");
      printf("** to estimate fault transmissibility multipliers.             **\n");
      printf("*****************************************************************\n");
    }
    
    /* MULTFLT */
    for (i=0; i < config_get_occurences(config , "MULTFLT"); i++) {
      const stringlist_type * tokens = config_iget_stringlist_ref(config , "MULTFLT" , i);
      const char * key         = stringlist_iget(tokens , 0);
      const char * ecl_file    = stringlist_iget(tokens , 1);
      const char * config_file = stringlist_iget(tokens , 2);
      
      ensemble_config_add_node(ensemble_config , key , PARAMETER , MULTFLT , ecl_file , NULL , multflt_config_fscanf_alloc(config_file));
    }
  }


  /* HAVANA_FAULT */
  for (i=0; i < config_get_occurences(config , "HAVANA_FAULT"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "HAVANA_FAULT" , i);
    const char * key         = stringlist_iget(tokens , 0);
    const char * config_file = stringlist_iget(tokens , 1);
    
    ensemble_config_add_node(ensemble_config , key , PARAMETER , HAVANA_FAULT , NULL , NULL , havana_fault_config_fscanf_alloc(config_file));
  }
  
  
  /* GEN_PARAM */
  for (i=0; i < config_get_occurences(config , "GEN_PARAM"); i++) {
    stringlist_type * tokens = config_iget_stringlist_ref(config , "GEN_PARAM" , i);
    char * key           = stringlist_iget_copy(tokens , 0);
    char * ecl_file      = stringlist_iget_copy(tokens , 1);
    stringlist_idel( tokens , 0 );
    stringlist_idel( tokens , 0 );
    {
      
      /* 
	 Required options:
	 * INPUT_FORMAT 
	 * INPUT_FILES
	 * INIT_FILES
	 * OUTPUT_FORMAT
	 
	 Optional:
	 * TEMPLATE
	 * KEY
	 */
      ensemble_config_add_node(ensemble_config , key , PARAMETER , GEN_DATA , ecl_file , NULL , gen_data_config_alloc(key , true , tokens , NULL , NULL));
    }
    free( key );
    free( ecl_file );
  }

  /* 
     For this datatype the cooperation between the enkf_node layer and
     the underlying type NOT particularly elegant.
     
     The problem is that the enkf_node layer owns the ECLIPSE
     input/output filenames. However, the node itself knows whether it
     should import/export ECLIPSE files (and therefore whether it
     needs the input/output filenames.
  */
  
  /* GEN_DATA */
  for (i=0; i < config_get_occurences(config , "GEN_DATA"); i++) {
    enkf_config_node_type * config_node;
    stringlist_type * tokens = config_iget_stringlist_ref(config , "GEN_DATA" , i);
    char * key               = stringlist_iget_copy(tokens , 0);
    gen_data_config_type * gen_data_config;
    {
      stringlist_idel(tokens , 0);
      {
	char * ecl_file;
	char * result_file;
	enkf_var_type var_type;
	gen_data_config = gen_data_config_alloc(key , false , tokens , &ecl_file , &result_file);
	if (ecl_file == NULL) /* 
				 EnKF should not provide the forward model with an instance of this
				 data => We have dynamic_result.
			      */
	  var_type = DYNAMIC_RESULT;
	else
	  var_type = DYNAMIC_STATE;   
	
	config_node = ensemble_config_add_node(ensemble_config , key , var_type , GEN_DATA , ecl_file , result_file , gen_data_config);
	
	util_safe_free( ecl_file );
	util_safe_free( result_file );
      }
    }
    {
      const gen_data_config_type * gen_data_config  = enkf_config_node_get_ref( config_node  );
      gen_data_type        * gen_data_min_std       = gen_data_config_get_min_std( gen_data_config );
      
      if (gen_data_min_std != NULL) {
        enkf_node_type * min_std_node = enkf_node_alloc_with_data( config_node , gen_data_min_std);
        enkf_config_node_set_min_std( config_node , min_std_node );
      }
    }
    free(key);
  }

  /* FIELD */
  {
    field_trans_table_type * field_trans_table = ensemble_config->field_trans_table;
    for (i=0; i < config_get_occurences(config , "FIELD"); i++) {
      stringlist_type * tokens = config_iget_stringlist_ref(config , "FIELD" , i);
      enkf_config_node_type * config_node = NULL;
      char *  key             = stringlist_iget_copy(tokens , 0);
      char *  var_type_string = stringlist_iget_copy(tokens , 1);
      stringlist_idel( tokens , 0 );   
      stringlist_idel( tokens , 0 );
      
      if (strcmp(var_type_string , "DYNAMIC") == 0) {
	config_node = ensemble_config_add_node(ensemble_config , key , DYNAMIC_STATE , FIELD , NULL , NULL , field_config_alloc_dynamic(key , grid , field_trans_table , tokens));
      } else if (strcmp(var_type_string , "PARAMETER") == 0) {
	char *  ecl_file        = stringlist_iget_copy(tokens , 0);
	stringlist_idel( tokens , 0 );
	
	config_node = ensemble_config_add_node(ensemble_config , key , PARAMETER   , FIELD , ecl_file , NULL , 
                                               field_config_alloc_parameter(key , ecl_file , grid , field_trans_table , tokens));
	free(ecl_file);
      } else if (strcmp(var_type_string , "GENERAL") == 0) {
	char * enkf_outfile = stringlist_iget_copy(tokens , 0); /* Out before in ?? */
	char * enkf_infile  = stringlist_iget_copy(tokens , 1);
	stringlist_idel( tokens , 0 );
	stringlist_idel( tokens , 0 );
	
	config_node = ensemble_config_add_node(ensemble_config , key , DYNAMIC_STATE , FIELD , enkf_outfile , enkf_infile , 
                                               field_config_alloc_general(key , enkf_outfile , grid , ecl_float_type , field_trans_table , tokens));
	free(enkf_outfile);
	free(enkf_infile);
      } else 
	util_abort("%s: FIELD type: %s is not recognized\n",__func__ , var_type_string);


      /**
         This will essentially install a min std instance.
      */
      {
        const field_config_type * field_config  = enkf_config_node_get_ref( config_node  );
        field_type        * field_min_std       = field_config_get_min_std( field_config );
        
        if (field_min_std != NULL) {
          enkf_node_type * min_std_node = enkf_node_alloc_with_data( config_node , field_min_std);
          enkf_config_node_set_min_std( config_node , min_std_node );
        }
      }
      
      free( key );
      free( var_type_string );
    }
  }

  /* GEN_KW */
  for (i=0; i < config_get_occurences(config , "GEN_KW"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "GEN_KW" , i);
    const char * key           = stringlist_iget(tokens , 0);
    const char * template_file = stringlist_iget(tokens , 1);
    const char * target_file   = stringlist_iget(tokens , 2);
    const char * config_file   = stringlist_iget(tokens , 3);
    const char * min_std_file  = NULL;
    enkf_config_node_type * config_node;

    if (stringlist_get_size( tokens ) > 4) 
      min_std_file = stringlist_iget( tokens , 4);
    
    config_node = ensemble_config_add_node(ensemble_config , key , PARAMETER , GEN_KW , target_file , NULL , 
                                           gen_kw_config_fscanf_alloc(key , config_file , template_file , min_std_file));
    {
      const gen_kw_config_type * gen_kw_config  = enkf_config_node_get_ref( config_node  );
      gen_kw_type              * gen_kw_min_std = gen_kw_config_get_min_std( gen_kw_config );

      if (gen_kw_min_std != NULL) {
        enkf_node_type * min_std_node = enkf_node_alloc_with_data( config_node , gen_kw_min_std);
        enkf_config_node_set_min_std( config_node , min_std_node );
      }
    }
      
  }

  /* SUMMARY */
  for (i=0; i < config_get_occurences(config , "SUMMARY"); i++) {
    const stringlist_type * tokens = config_iget_stringlist_ref(config , "SUMMARY" , i);
    const char * key        = stringlist_iget(tokens , 0);
    
    ensemble_config_ensure_summary(ensemble_config , key );
  }
  
  /*****************************************************************/

    
  return ensemble_config;
}

/**
   This function takes a string like this: "PRESSURE:1,4,7" - it
   splits the string on ":" and tries to lookup a config object with
   that key. For the general string A:B:C:D it will try consecutively
   the keys: A, A:B, A:B:C, A:B:C:D. If a config object is found it is
   returned, otherwise NULL is returned.

   The last argument is the pointer to a string which will be updated
   with the node-spesific part of the full key. So for instance with
   the example "PRESSURE:1,4,7", the index_key will contain
   "1,4,7". If the full full_key is used to find an object index_key
   will be NULL, that also applies if no object is found.
*/

   

const enkf_config_node_type * ensemble_config_user_get_node(const ensemble_config_type * config , const char  * full_key, char ** index_key ) {
  const enkf_config_node_type * node = NULL;
  char ** key_list;
  int     keys;
  int     key_length = 1;
  int offset;
  
  *index_key = NULL;
  util_split_string(full_key , ":" , &keys , &key_list);
  while (node == NULL && key_length <= keys) {
    char * current_key = util_alloc_joined_string( (const char **) key_list , key_length , ":");
    if (ensemble_config_has_key(config , current_key))
      node = ensemble_config_get_node(config , current_key);
    else
      key_length++;
    offset = strlen( current_key );
    free( current_key );
  }
  if (node != NULL) {
    if (offset < strlen( full_key ))
      *index_key = util_alloc_string_copy(&full_key[offset+1]);
  }
  
  util_free_stringlist(key_list , keys);
  return node;
}



stringlist_type * ensemble_config_alloc_keylist(const ensemble_config_type * config) {
  return hash_alloc_stringlist( config->config_nodes );
}


/**
   Observe that var_type here is an integer - naturally written as a
   sum of enkf_var_type values:

     ensemble_config_alloc_keylist_from_var_type( config , PARAMETER + DYNAMIC_STATE);
*/
   
stringlist_type * ensemble_config_alloc_keylist_from_var_type(const ensemble_config_type * config , int var_type) {
  stringlist_type * key_list = stringlist_alloc_new();
  hash_iter_type * iter = hash_iter_alloc(config->config_nodes);
  const char * key = hash_iter_get_next_key(iter);
  while (key != NULL) {
    if (enkf_config_node_get_var_type( hash_get(config->config_nodes , key)) & var_type)
      stringlist_append_copy( key_list , key );
    
    key = hash_iter_get_next_key(iter);
  }
  hash_iter_free(iter);
  return key_list;
}



stringlist_type * ensemble_config_alloc_keylist_from_impl_type(const ensemble_config_type * config , enkf_impl_type impl_type) {
  stringlist_type * key_list = stringlist_alloc_new();
  hash_iter_type * iter = hash_iter_alloc(config->config_nodes);
  const char * key = hash_iter_get_next_key(iter);
  while (key != NULL) {
    if (enkf_config_node_get_impl_type( hash_get(config->config_nodes , key)) == impl_type)
      stringlist_append_copy( key_list , key );
    
    key = hash_iter_get_next_key(iter);
  }
  hash_iter_free(iter);
  return key_list;
}




void ensemble_config_init_internalization( ensemble_config_type * config ) {
  hash_iter_type * iter = hash_iter_alloc(config->config_nodes);
  const char * key = hash_iter_get_next_key(iter);
  while (key != NULL) {
    enkf_config_node_init_internalization( hash_get(config->config_nodes , key));
    key = hash_iter_get_next_key(iter);
  }
  hash_iter_free(iter);
}




