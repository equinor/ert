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
#include <meas_data.h>
#include <enkf_types.h>
#include <sched_file.h>
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
#include <gen_kw_config.h>
#include <gen_data_config.h>
#include <stringlist.h>
#include <ensemble_config.h>
#include <config.h>
#include <gen_data_config.h>
#include <pthread.h>                /* Must have rw locking on the config_nodes ... */
#include <field_trans.h>
#include <subst_func.h>
#include <enkf_obs.h>
#include <ecl_config.h>
#include "config_keys.h"
#include "enkf_defaults.h"


struct ensemble_config_struct {
  pthread_mutex_t          mutex;
  char                   * gen_kw_format_string;   /* Format string used when creating gen_kw search/replace strings. */
  hash_type              * config_nodes;           /* A hash of enkf_config_node instances - which again conatin pointers to e.g. field_config objects.  */
  field_trans_table_type * field_trans_table;      /* A table of the transformations which are available to apply on fields. */
  const ecl_sum_type     * refcase;                /* A ecl_sum reference instance - can be NULL (NOT owned by the ensemble
                                                      config). Is only used to check that summary keys are valid when adding. */
};



/**
   Setting the format string used to 'mangle' the string in the gen_kw
   template files. Consider the following example:

      Parameter file
      --------------
      MULTPV   LOGUNIF  0.0001 0.10


      Template file
      -------------
      BOX
         1  10  1 10  1 5 /

      MULTPV  500*__MULTPV__

   Here the parameter file defines a parameter named 'MULTPV', and the
   template file uses the marker string '__MULTPV__' which should be
   replaced with a numerical value. For the current example the
   gen_kw_format_string should have the value '__%s__'.

   There are no rules for the format string, but it _must_ contain a
   '%s' placeholder which will be replaced with the parameter name
   (this is not checked for). The function call creating a search
   string from a parameter name is:

      tagged_string = util_alloc_sprintf( gen_kw_format_string , parameter_name );

*/

void ensemble_config_set_gen_kw_format( ensemble_config_type * ensemble_config , const char * gen_kw_format_string) {
  if (!util_string_equal( gen_kw_format_string , ensemble_config->gen_kw_format_string)) {
    stringlist_type * gen_kw_keys = ensemble_config_alloc_keylist_from_impl_type( ensemble_config , GEN_KW );
    int i;
    ensemble_config->gen_kw_format_string = util_realloc_string_copy( ensemble_config->gen_kw_format_string , gen_kw_format_string );
    for (i=0; i < stringlist_get_size( gen_kw_keys ); i++) {
      enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , stringlist_iget( gen_kw_keys , i ));
      gen_kw_config_update_tag_format( enkf_config_node_get_ref( config_node ) , gen_kw_format_string );
    }
    stringlist_free( gen_kw_keys );
  }
}


const char * ensemble_config_get_gen_kw_format( const ensemble_config_type * ensemble_config ) {
  return ensemble_config->gen_kw_format_string;
}


void ensemble_config_set_refcase( ensemble_config_type * ensemble_config , const ecl_sum_type * refcase) {
  ensemble_config->refcase = refcase;
}
                                                                                                           



ensemble_config_type * ensemble_config_alloc_empty( ) {

  ensemble_config_type * ensemble_config = util_malloc(sizeof * ensemble_config , __func__);
  ensemble_config->config_nodes          = hash_alloc();
  ensemble_config->refcase               = NULL;
  ensemble_config->gen_kw_format_string  = util_alloc_string_copy( DEFAULT_GEN_KW_TAG_FORMAT );
  pthread_mutex_init( &ensemble_config->mutex , NULL);
  
  return ensemble_config;
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
  hash_free( ensemble_config->config_nodes );
  field_trans_table_free( ensemble_config->field_trans_table );
  free( ensemble_config->gen_kw_format_string );
  free( ensemble_config );
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


void ensemble_config_add_node__( ensemble_config_type * ensemble_config , enkf_config_node_type * node) {
  
  const char * key = enkf_config_node_get_key( node );
  if (ensemble_config_has_key(ensemble_config , key)) 
    util_abort("%s: a configuration object:%s has already been added - aborting \n",__func__ , key);
  hash_insert_hash_owned_ref(ensemble_config->config_nodes , key , node , enkf_config_node_free__);
}


enkf_config_node_type *  ensemble_config_add_node(ensemble_config_type * ensemble_config , 
                                                  const char    * key              , 
                                                  enkf_var_type  enkf_type         , 
                                                  enkf_impl_type impl_type         ,
                                                  const char   * enkf_outfile      , /* Written by EnKF and read by forward model */
                                                  const char   * enkf_infile       , /* Written by forward model and read by EnKF */ 
                                                  void         * data ) {

    if (ensemble_config_has_key(ensemble_config , key)) 
    util_abort("%s: a configuration object:%s has already been added - aborting \n",__func__ , key);
  
  {
    enkf_config_node_type * node = enkf_config_node_alloc(enkf_type , impl_type , key , enkf_outfile , enkf_infile , data );
    hash_insert_hash_owned_ref(ensemble_config->config_nodes , key , node , enkf_config_node_free__);
    return node;
  }
}



/**
   This is called by the enkf_state function while loading results,
   that code is run in parallell by many threads.
*/
void ensemble_config_ensure_static_key(ensemble_config_type * ensemble_config , const char * kw ) {
  pthread_mutex_lock( &ensemble_config->mutex );
  {
    if (!ensemble_config_has_key(ensemble_config , kw)) 
      ensemble_config_add_node(ensemble_config , kw , STATIC_STATE , STATIC , NULL , NULL , NULL);
  }
  pthread_mutex_unlock( &ensemble_config->mutex );
}


///* 
//   Required options:
//   * INPUT_FORMAT 
//   * INPUT_FILES
//   * INIT_FILES
//   * OUTPUT_FORMAT
//       
//   Optional:
//   * TEMPLATE
//   * KEY
//*/
//
//void ensemble_config_add_gen_param(ensemble_config_type * config , const char * key , const char * enkf_outfile , stringlist_type * options) {
//  gen_data_config_type * node = gen_data_config_alloc_with_options( key , true , options );
//  {
//    //char                  * enkf_outfile   = gen_data_config_pop_enkf_outfile( node );
//    enkf_config_node_type * config_node      = ensemble_config_add_node( config , key , PARAMETER , GEN_DATA , enkf_outfile , NULL , node );
//    gen_data_type         * gen_data_min_std = gen_data_config_get_min_std( node );
//
//    if (gen_data_min_std != NULL) {
//      enkf_node_type * min_std_node = enkf_node_alloc_with_data( config_node , gen_data_min_std);
//      enkf_config_node_set_min_std( config_node , min_std_node );
//    }
//
//    //util_safe_free( enkf_outfile );
//  }
//}
//
//
//
//
///* 
//   For this datatype the cooperation between the enkf_node layer and
//   the underlying type NOT particularly elegant.
//   
//   The problem is that the enkf_node layer owns the ECLIPSE
//   input/output filenames. However, the node itself knows whether it
//   should import/export ECLIPSE files (and therefore whether it
//   needs the input/output filenames.
//*/
//
//
//void ensemble_config_add_gen_data(ensemble_config_type * config , const char * key , stringlist_type * options) {
//  enkf_var_type var_type;
//  char * enkf_outfile;
//  char * enkf_infile;
//  gen_data_config_type * node = gen_data_config_alloc_with_options( key , false , options);
//  enkf_outfile = gen_data_config_pop_enkf_outfile( node );
//  enkf_infile  = gen_data_config_pop_enkf_infile( node );
//  
//  if (enkf_outfile == NULL) 
//    /* 
//       EnKF should not provide the forward model with an instance of this
//       data => We have dynamic_result.
//    */
//    var_type = DYNAMIC_RESULT;
//  else
//    var_type = DYNAMIC_STATE;   
//
//  {
//    enkf_config_node_type * config_node      = ensemble_config_add_node( config , key , var_type , GEN_DATA , enkf_outfile , enkf_infile , node );
//    gen_data_type         * gen_data_min_std = gen_data_config_get_min_std( node );
//    
//    if (gen_data_min_std != NULL) {
//      enkf_node_type * min_std_node = enkf_node_alloc_with_data( config_node , gen_data_min_std);
//      enkf_config_node_set_min_std( config_node , min_std_node );
//    }
//  }
//  
//  util_safe_free( enkf_outfile );
//  util_safe_free( enkf_infile );
//}








void ensemble_config_add_obs_key(ensemble_config_type * ensemble_config , const char * key, const char * obs_key) {
  enkf_config_node_type * config_node = hash_get(ensemble_config->config_nodes , key);
  enkf_config_node_add_obs_key(config_node , obs_key);
}


void ensemble_config_clear_obs_keys(ensemble_config_type * ensemble_config) {
  hash_iter_type * iter = hash_iter_alloc( ensemble_config->config_nodes );
  while (!hash_iter_is_complete( iter )) {
    enkf_config_node_type * config_node = hash_iter_get_next_value( iter );
    enkf_config_node_clear_obs_keys( config_node );
  }
  hash_iter_free( iter );
}



void ensemble_config_add_config_items(config_type * config) {
  config_item_type * item;

  /** 
      The two fault types are just added to the CONFIG object only to
      be able to print suitable messages before exiting.
  */
      
  item = config_add_item(config , "HAVANA_FAULT" , false , true);
  config_item_set_argc_minmax(item , 2 , 2 ,  0 , NULL );

  item = config_add_item(config , "MULTFLT" , false , true);
  config_item_set_argc_minmax(item , 3 , 3 ,  3 , (const config_item_types [3]) { CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});


  /*****************************************************************/
  
  item = config_add_item(config , GEN_KW_KEY , false , true);
  config_item_set_argc_minmax(item , 4 , 6 ,  6 , (const config_item_types [6]) { CONFIG_STRING , CONFIG_EXISTING_FILE , CONFIG_STRING , CONFIG_EXISTING_FILE , CONFIG_STRING , CONFIG_STRING});
  
  item = config_add_key_value( config , GEN_KW_TAG_FORMAT_KEY , false , CONFIG_STRING);
  
  item = config_add_item(config , SCHEDULE_PREDICTION_FILE_KEY , false , false);
  /* SCEDHULE_PREDICTION_FILE   FILENAME  <PARAMETERS:> <INIT_FILES:> */
  config_item_set_argc_minmax(item , 1 , 3 ,  3 , (const config_item_types [3]) { CONFIG_EXISTING_FILE , CONFIG_STRING , CONFIG_STRING});

  item = config_add_item(config , GEN_PARAM_KEY , false , true);
  config_item_set_argc_minmax(item , 5 , -1 ,  0 , NULL);
  
  item = config_add_item(config , GEN_DATA_KEY , false , true);
  config_item_set_argc_minmax(item , 1 , -1 ,  0 , NULL);

  item = config_add_item(config , SUMMARY_KEY , false , true);   /* Can have several summary keys on each line. */
  config_item_set_argc_minmax(item , 1 , -1 ,  0 , NULL);
  
  /* 
     The way config info is entered for fields is unfortunate because
     it is difficult/impossible to let the config system handle run
     time validation of the input.
  */
  
  item = config_add_item(config , FIELD_KEY , false , true);
  config_item_set_argc_minmax(item , 2 , -1 ,  0 , NULL);
  config_item_add_required_children(item , GRID_KEY);   /* If you are using a FIELD - you must have a grid. */
}


/**
   Observe that if the user has not given a refcase with the REFCASE
   key the refcase pointer will be NULL. In that case it will be
   impossible to use wildcards when expanding summary variables.
*/

void ensemble_config_init(ensemble_config_type * ensemble_config , const config_type * config , ecl_grid_type * grid, const ecl_sum_type * refcase) {
  int i;
  ensemble_config->field_trans_table     = field_trans_table_alloc();    
  ensemble_config_set_refcase( ensemble_config , refcase );

  /* MULTFLT depreceation warning added 17/03/09 (svn 1811). */
  if (config_get_occurences(config , "MULTFLT") > 0) {
    printf("******************************************************************\n");
    printf("** You have used the keyword MULTFLT - this is unfortunately no **\n");
    printf("** longer supported - use GEN_KW instead.                       **\n");
    printf("******************************************************************\n");
    exit(1);
  }

  if (config_get_occurences(config , "HAVANA_FAULT") > 0) {
    printf("************************************************************************\n");
    printf("** You have used the keyword HAVANA_FAULT - this is unfortunately no  **\n");
    printf("** longer supported - use GEN_KW instead and a suitable FORWARD_MODEL.**\n");
    printf("************************************************************************\n");
    exit(1);
  }

  if (config_item_set( config , GEN_KW_TAG_FORMAT_KEY))
    ensemble_config_set_gen_kw_format( ensemble_config , config_iget( config , GEN_KW_TAG_FORMAT_KEY , 0 , 0 ));
  
  /* GEN_PARAM  - should be unified with the GEN_DATA*/
  for (i=0; i < config_get_occurences(config , GEN_PARAM_KEY); i++) {
    stringlist_type * tokens = config_iget_stringlist_ref(config , GEN_PARAM_KEY , i);
    const char * key                          = stringlist_iget(tokens , 0);
    const char * ecl_file                     = stringlist_iget(tokens , 1);  /* Only difference from GEN_DATA is that the ECL_FILE is not a ":" keyword. */
    enkf_config_node_type * config_node       = ensemble_config_add_gen_data( ensemble_config , key );
    {
      hash_type * options = hash_alloc_from_options( tokens );
      gen_data_file_format_type input_format  = gen_data_config_check_format( hash_safe_get( options , INPUT_FORMAT_KEY));
      gen_data_file_format_type output_format = gen_data_config_check_format( hash_safe_get( options , OUTPUT_FORMAT_KEY));
      const char * init_file_fmt              = hash_safe_get( options , INIT_FILES_KEY);
      const char * template                   = hash_safe_get( options , TEMPLATE_KEY);
      const char * key                        = hash_safe_get( options , KEY_KEY);
      const char * result_file                = hash_safe_get( options , RESULT_FILE_KEY);
      const char * min_std_file               = hash_safe_get( options , MIN_STD_KEY);
      
      enkf_config_node_update_gen_data( config_node , input_format , output_format , init_file_fmt , template , key , ecl_file , result_file , min_std_file);
      
      hash_free( options );
    }
  }
  
  /* GEN_DATA */
  for (i=0; i < config_get_occurences(config , GEN_DATA_KEY); i++) {
    stringlist_type * tokens = config_iget_stringlist_ref(config , GEN_DATA_KEY , i);
    const char * key                          = stringlist_iget(tokens , 0);
    enkf_config_node_type * config_node       = ensemble_config_add_gen_data( ensemble_config , key );
    {
      hash_type * options = hash_alloc_from_options( tokens );
      gen_data_file_format_type input_format  = gen_data_config_check_format( hash_safe_get( options , INPUT_FORMAT_KEY));
      gen_data_file_format_type output_format = gen_data_config_check_format( hash_safe_get( options , OUTPUT_FORMAT_KEY));
      const char * init_file_fmt              = hash_safe_get( options , INIT_FILES_KEY);
      const char * template                   = hash_safe_get( options , TEMPLATE_KEY);
      const char * key                        = hash_safe_get( options , KEY_KEY);
      const char * ecl_file                   = hash_safe_get( options , ECL_FILE_KEY);
      const char * result_file                = hash_safe_get( options , RESULT_FILE_KEY);
      const char * min_std_file               = hash_safe_get( options , MIN_STD_KEY);


      enkf_config_node_update_gen_data( config_node , input_format , output_format , init_file_fmt , template , key , ecl_file , result_file , min_std_file);
      hash_free( options );
    }
  }


  /* FIELD */
  {
    for (i=0; i < config_get_occurences(config , FIELD_KEY); i++) {
      stringlist_type * tokens            = config_iget_stringlist_ref(config , FIELD_KEY , i);
      const char *  key                   = stringlist_iget(tokens , 0);
      const char *  var_type_string       = stringlist_iget(tokens , 1);
      enkf_config_node_type * config_node = ensemble_config_add_field( ensemble_config , key , grid );
      
      {
        hash_type * options = hash_alloc_from_options( tokens );
        
        int    truncation = TRUNCATE_NONE;
        double value_min  = -1;
        double value_max  = -1;
        
        if (hash_has_key( options , MIN_KEY)) {
          truncation |= TRUNCATE_MIN;
          value_min   = atof(hash_get( options , MIN_KEY));
        }

        if (hash_has_key( options , MAX_KEY)) {
          truncation |= TRUNCATE_MAX;
          value_max   = atof(hash_get( options , MAX_KEY));
        }
        
        
        if (strcmp(var_type_string , DYNAMIC_KEY) == 0) 
          enkf_config_node_update_state_field( config_node , truncation , value_min , value_max );
        else if (strcmp(var_type_string , PARAMETER_KEY) == 0) {
          const char *  ecl_file          = stringlist_iget(tokens , 2);
          const char *  init_file_fmt     = hash_safe_get( options , INIT_FILES_KEY );
          const char *  init_transform    = hash_safe_get( options , INIT_TRANSFORM_KEY );
          const char *  output_transform  = hash_safe_get( options , OUTPUT_TRANSFORM_KEY );
          const char *  min_std_file      = hash_safe_get( options , MIN_STD_KEY);
          
          enkf_config_node_update_parameter_field( config_node, 
                                                   ecl_file          , 
                                                   init_file_fmt     , 
                                                   min_std_file      , 
                                                   truncation        , 
                                                   value_min         , 
                                                   value_max         ,    
                                                   init_transform    , 
                                                   output_transform   );
        } else if (strcmp(var_type_string , GENERAL_KEY) == 0) {
          const char *  ecl_file          = stringlist_iget(tokens , 2);
          const char *  enkf_infile       = stringlist_iget(tokens , 3);
          const char *  init_file_fmt     = hash_safe_get( options , INIT_FILES_KEY );
          const char *  init_transform    = hash_safe_get( options , INIT_TRANSFORM_KEY );
          const char *  output_transform  = hash_safe_get( options , OUTPUT_TRANSFORM_KEY );
          const char *  input_transform   = hash_safe_get( options , INPUT_TRANSFORM_KEY );
          const char *  min_std_file      = hash_safe_get( options , MIN_STD_KEY);
          

          enkf_config_node_update_general_field( config_node,
                                                 ecl_file , 
                                                 enkf_infile , 
                                                 init_file_fmt , 
                                                 min_std_file , 
                                                 truncation , value_min , value_max , 
                                                 init_transform , 
                                                 input_transform , 
                                                 output_transform);

          
        } else 
          util_abort("%s: FIELD type: %s is not recognized\n",__func__ , var_type_string);
        
        hash_free( options );
      }
    }
  }

  /* GEN_KW */
  for (i=0; i < config_get_occurences(config , GEN_KW_KEY); i++) {
    stringlist_type * tokens = config_iget_stringlist_ref(config , GEN_KW_KEY , i);
    char * key            = stringlist_iget_copy(tokens , 0);
    char * template_file  = stringlist_iget_copy(tokens , 1);
    char * enkf_outfile   = stringlist_iget_copy(tokens , 2);
    char * parameter_file = stringlist_iget_copy(tokens , 3);
    stringlist_idel( tokens , 0);
    stringlist_idel( tokens , 0);
    stringlist_idel( tokens , 0);
    stringlist_idel( tokens , 0);

    {
      hash_type * opt_hash                = hash_alloc_from_options( tokens );
      enkf_config_node_type * config_node = ensemble_config_add_gen_kw( ensemble_config , key );
      enkf_config_node_update_gen_kw( config_node , 
                                      enkf_outfile , 
                                      template_file , 
                                      parameter_file , 
                                      hash_safe_get( opt_hash , MIN_STD_KEY ) , 
                                      hash_safe_get( opt_hash , INIT_FILES_KEY));
      hash_free( opt_hash );
    }
  
    free(key);
    free(template_file);
    free(enkf_outfile);
    free(parameter_file);
  }


  /* SUMMARY */
  {
    stringlist_type * keys = stringlist_alloc_new ( );
    
    for (i=0; i < config_get_occurences(config , SUMMARY_KEY ); i++) {
      int j,k;
      const stringlist_type * summary_kw_list = config_iget_stringlist_ref(config , SUMMARY_KEY , i);
      for (j= 0; j < stringlist_get_size( summary_kw_list ); j++) {
        const char * key = stringlist_iget( summary_kw_list , j); 
        
        if (util_string_has_wildcard( key )) {
          if (ensemble_config->refcase != NULL) {
            ecl_sum_select_matching_general_var_list( ensemble_config->refcase , key , keys );   /* Expanding the wildcard notatition with help of the refcase. */
            for (k=0; k < stringlist_get_size( keys ); k++) 
              ensemble_config_add_summary(ensemble_config , stringlist_iget(keys , k) );
          } else
            util_exit("ERROR: When using SUMMARY wildcards like: \"%s\" you must supply a valid refcase.\n",key);
        } else 
          ensemble_config_add_summary(ensemble_config , key );
      }
    }
    
    stringlist_free( keys );
  }
  
  /*****************************************************************/
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
  util_split_string(full_key , USER_KEY_JOIN_STRING , &keys , &key_list);
  while (node == NULL && key_length <= keys) {
    char * current_key = util_alloc_joined_string( (const char **) key_list , key_length , USER_KEY_JOIN_STRING );
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
   
stringlist_type * ensemble_config_alloc_keylist_from_var_type(const ensemble_config_type * config , int var_mask) {
  stringlist_type * key_list = stringlist_alloc_new();
  hash_iter_type * iter = hash_iter_alloc(config->config_nodes);

  while (!hash_iter_is_complete( iter )) {
    const char * key       = hash_iter_get_next_key(iter);
    enkf_var_type var_type = enkf_config_node_get_var_type( hash_get(config->config_nodes , key));
    
    if (var_type & var_mask)
      stringlist_append_copy( key_list , key );
  }
  hash_iter_free(iter);

  return key_list;
}



stringlist_type * ensemble_config_alloc_keylist_from_impl_type(const ensemble_config_type * config , enkf_impl_type impl_type) {
  stringlist_type * key_list = stringlist_alloc_new();
  hash_iter_type * iter = hash_iter_alloc(config->config_nodes);
  while (!hash_iter_is_complete( iter )) {
    const char * key = hash_iter_get_next_key(iter);
    if (enkf_config_node_get_impl_type( hash_get(config->config_nodes , key)) == impl_type)
      stringlist_append_copy( key_list , key );

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




/**
   This function will look up the user_key in the ensemble_config. If
   the corresponding config_node can not be found 0 will be returned,
   otherwise enkf_config_node functions will be invoked.
*/


int ensemble_config_get_observations( const ensemble_config_type * config , enkf_obs_type * enkf_obs , const char * user_key , int obs_count , time_t * obs_time , double * y , double * std) {
  int num_obs = 0;
  char * index_key;
  const enkf_config_node_type * config_node = ensemble_config_user_get_node( config , user_key , &index_key);
  if (config_node != NULL) {
    num_obs = enkf_config_node_load_obs( config_node , enkf_obs , index_key , obs_count , obs_time , y , std);
    util_safe_free( index_key );
  } 
  return num_obs;
}


/*****************************************************************/


/* 
   The ensemble_config_add_xxx() functions below will create a new xxx
   instance and add it to the ensemble_config; the return value from
   the functions is the newly created config_node instances.

   The newly created enkf_config_node instances are __NOT__ fully
   initialized, and a subsequent call to enkf_config_node_update_xxx()
   is essential for proper operation.
*/

enkf_config_node_type * ensemble_config_add_field( ensemble_config_type * config , const char * key , ecl_grid_type * ecl_grid ) {
  enkf_config_node_type * config_node = enkf_config_node_new_field( key , ecl_grid , config->field_trans_table );
  ensemble_config_add_node__( config , config_node );
  return config_node;
}


enkf_config_node_type * ensemble_config_add_gen_kw( ensemble_config_type * config , const char * key ) {
  enkf_config_node_type * config_node = enkf_config_node_new_gen_kw( key , config->gen_kw_format_string );
  ensemble_config_add_node__( config , config_node );
  return config_node;
}


enkf_config_node_type * ensemble_config_add_gen_data( ensemble_config_type * config , const char * key ) {
  enkf_config_node_type * config_node = enkf_config_node_new_gen_data( key );
  ensemble_config_add_node__( config , config_node );
  return config_node;
}


/**
   This function ensures that object contains a node with 'key' and
   type == SUMMARY.
   
   If the @refcase pointer is different from NULL the key will be
   validated. Keys which do not exist in the refcase will be ignored,
   a warning will be printed on stderr and the function will return
   NULL.
*/

enkf_config_node_type * ensemble_config_add_summary(ensemble_config_type * ensemble_config , const char * key) {
  enkf_config_node_type * config_node = NULL;
  if (hash_has_key(ensemble_config->config_nodes, key)) {
    if (ensemble_config_impl_type(ensemble_config , key) != SUMMARY)
      util_abort("%s: ensemble key:%s already existst - but it is not of summary type\n",__func__ , key);
  } else {
    if ((ensemble_config->refcase == NULL) || (ecl_sum_has_general_var( ensemble_config->refcase , key ))) {
      config_node = enkf_config_node_alloc_summary( key );
      ensemble_config_add_node__(ensemble_config , config_node );
    } else
      fprintf(stderr,"** Warning: the refcase:%s does not contain the summary key:\"%s\" - will be ignored.\n", ecl_sum_get_case( ensemble_config->refcase ) , key);
  }
  return config_node;
}


/*****************************************************************/

void ensemble_config_fprintf_config( ensemble_config_type * ensemble_config , FILE * stream ) {
  fprintf( stream , CONFIG_COMMENTLINE_FORMAT );
  fprintf( stream , CONFIG_COMMENT_FORMAT , "Here comes configuration information about the uncertain parameters and response variables in use.");

  fprintf( stream , CONFIG_KEY_FORMAT      , GEN_KW_TAG_FORMAT_KEY );
  fprintf( stream , CONFIG_ENDVALUE_FORMAT , ensemble_config->gen_kw_format_string);


  /* Writing GEN_KW nodes. */
  {
    stringlist_type * gen_kw_keys = ensemble_config_alloc_keylist_from_impl_type( ensemble_config , GEN_KW );
    stringlist_sort( gen_kw_keys , NULL );
    for (int i=0; i < stringlist_get_size( gen_kw_keys ); i++) {
      const enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , stringlist_iget( gen_kw_keys , i));
      enkf_config_node_fprintf_config( config_node , stream );
    }
    if (stringlist_get_size( gen_kw_keys ) > 0)
      fprintf(stream , "\n");
    stringlist_free( gen_kw_keys );
  }

  
  /* Writing FIELD nodes. */
  {
    stringlist_type * field_keys = ensemble_config_alloc_keylist_from_impl_type( ensemble_config , FIELD );
    stringlist_sort( field_keys , NULL );
    for (int i=0; i < stringlist_get_size( field_keys ); i++) {
      const enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , stringlist_iget( field_keys , i));
      enkf_config_node_fprintf_config( config_node , stream );
    }
    if (stringlist_get_size( field_keys ) > 0)
      fprintf(stream , "\n");
    stringlist_free( field_keys );
  }


  /* Writing SUMMARY nodes. */
  {
    stringlist_type * summary_keys = ensemble_config_alloc_keylist_from_impl_type( ensemble_config , SUMMARY );
    stringlist_sort( summary_keys , NULL );
    for (int i=0; i < stringlist_get_size( summary_keys ); i++) {
      if (i == 0)
        fprintf(stream , CONFIG_KEY_FORMAT , SUMMARY_KEY);
      else if ((i % 8) == 0) {
        fprintf(stream , "\n");
        fprintf(stream , CONFIG_KEY_FORMAT , SUMMARY_KEY);
      }
      fprintf(stream , CONFIG_SHORT_VALUE_FORMAT , stringlist_iget( summary_keys , i ));
    }
    fprintf(stream , "\n");
    stringlist_free( summary_keys );
  }
  fprintf(stream , "\n");
  

  /* Writing GEN_DATA nodes. */
  {
    stringlist_type * gen_data_keys = ensemble_config_alloc_keylist_from_impl_type( ensemble_config , GEN_DATA );
    stringlist_sort( gen_data_keys , NULL );
    for (int i=0; i < stringlist_get_size( gen_data_keys ); i++) {
      const enkf_config_node_type * config_node = ensemble_config_get_node( ensemble_config , stringlist_iget( gen_data_keys , i));
      enkf_config_node_fprintf_config( config_node , stream );
    }
    stringlist_free( gen_data_keys );
  }
  fprintf(stream , "\n\n");
}
