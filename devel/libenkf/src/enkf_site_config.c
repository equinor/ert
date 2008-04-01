#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <util.h>
#include <hash.h>
#include <enkf_site_config.h>

enkf_site_config_node_type * enkf_site_config_get_node(const enkf_site_config_type * , const char * );
static bool                  enkf_site_config_assert_set_int(const enkf_site_config_type * , const enkf_site_config_node_type *  );

/*****************************************************************/

typedef bool   (validate_ftype) (const enkf_site_config_type * , const enkf_site_config_node_type *);


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


void enkf_site_config_node_set_value(enkf_site_config_node_type * node , const char * value) {
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
  printf("Setting: %s->%s \n",node->key,value);
}


const char * enkf_site_config_node_get_value(const enkf_site_config_node_type * node) {
  return node->value;
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
    enkf_site_config_node_set_value(node , default_value);
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


static bool enkf_site_config_validate_queue_system(const enkf_site_config_type * site , const enkf_site_config_node_type * node) {
  bool valid = true;
  const char * queue_system = enkf_site_config_node_get_value(node);
  enkf_site_config_node_type * max_running_node;
  if (strcmp(queue_system , "LSF") == 0) {
    if ( !(enkf_site_config_node_set(site , "LSF_QUEUE") && enkf_site_config_node_set(site , "LSF_RESOURCES")) ) {
      fprintf(stderr,"** When using LSF as queue system, you must specify LSF_QUEUE and LSF_RESOURCES.\n");
      valid = false;
    }
    max_running_node = enkf_site_config_get_node(site , "MAX_RUNNING_LSF");
  } else if (strcmp(queue_system , "LOCAL") == 0) {
    max_running_node = enkf_site_config_get_node(site , "MAX_RUNNING_LOCAL");
  } else if (strcmp(queue_system , "RSH") == 0) {
    max_running_node = enkf_site_config_get_node(site , "MAX_RUNNING_RSH");
    if (!enkf_site_config_node_set(site , "RSH_HOST_LIST")) {
      fprintf(stderr , " ** Must set key RSH_HOST_LIST when using the RSH queue driver\n");
      valid = false;
    }
    if (!enkf_site_config_node_set(site , "RSH_COMMAND")) {
      fprintf(stderr , " ** Must set key RSH_COMMAND when using the RSH queue driver\n");
      valid = false;
    }
  } else {
    fprintf(stderr,"%s: queue_system:%s not recognized - serious internal error - aborting \n",__func__ , queue_system);
    abort();
  }
  
  valid = (valid && enkf_site_config_assert_set_int(site , max_running_node));
  return valid;
}


static bool enkf_site_config_validate_queue_name(const enkf_site_config_type * config , const enkf_site_config_node_type * node) {
  /* Could call the lsf_driver to validate this queue_name .... */
  return true;
}

static bool enkf_site_config_assert_set(const enkf_site_config_type * config , const enkf_site_config_node_type *  node) {
  if (!node->value_set)
    fprintf(stderr,"** You must supply a value for the %s key.\n",node->key);
  return node->value_set;
}


static bool enkf_site_config_assert_set_int(const enkf_site_config_type * config , const enkf_site_config_node_type *  node) {
  bool valid = enkf_site_config_assert_set(config , node);
  if (valid) {
    int dummy;
    if (!util_sscanf_int(node->value , &dummy)) {
      fprintf(stderr," ** Failed to convert %s=%s to an integer.\n",node->key , node->value);
      valid = false;
    }
  }
  return valid;
}


static bool enkf_site_config_assert_set_executable(const enkf_site_config_type * config , const enkf_site_config_node_type * node) {
  bool valid = false;
  if (enkf_site_config_assert_set(config , node)) {
    if (util_file_exists(node->value))
      if (util_is_executable(node->value))
	valid = true;

    if (!valid) 
      fprintf(stderr,"** %s must point to an existing executable file.\n",node->key);
    
  }
  return valid;
}

static bool enkf_site_config_assert_set_existing(const enkf_site_config_type * config , const enkf_site_config_node_type * node) {
  bool valid = false;
  if (enkf_site_config_assert_set(config , node)) {
    if (util_file_exists(node->value))
      valid = true;
    
    if (!valid) 
      fprintf(stderr,"** %s must point to an existing file.\n",node->key);
  }
  return valid;
}

static bool enkf_site_config_assert_eclipse_executable(const enkf_site_config_type * config , const enkf_site_config_node_type * node) {
  return enkf_site_config_assert_set(config , node);
}

/*****************************************************************/

enkf_site_config_node_type * enkf_site_config_get_node(const enkf_site_config_type * site , const char * key) {
  return hash_get(site->config , key);
}


const char * enkf_site_config_get_value(const enkf_site_config_type * site , const char * key) {
  enkf_site_config_node_type * node = enkf_site_config_get_node(site , key);
  return node->value;
}


bool enkf_site_config_node_set(const enkf_site_config_type * site , const char * key) {
  enkf_site_config_node_type * node = enkf_site_config_get_node(site , key);
  return node->value_set;
}


void enkf_site_config_add_node(enkf_site_config_type * site , const char * key , const char * default_value , int selection_size , const char ** selection_set , validate_ftype * validate) {
  hash_insert_hash_owned_ref(site->config , key , enkf_site_config_node_alloc(key , default_value , selection_size , selection_set , validate) , enkf_site_config_node_free__);
}


bool enkf_site_config_has_key(const enkf_site_config_type * site ,const char * key) {
  return hash_has_key(site->config , key);
}


void enkf_site_config_set_key(enkf_site_config_type * site , const char * key , const char * value) {
  if (enkf_site_config_has_key(site , key)) {
    enkf_site_config_node_type * node = enkf_site_config_get_node(site , key);
    enkf_site_config_node_set_value(node , value);
  } else {
    fprintf(stderr,"%s: key:%s is not recognized - aborting \n",__func__ , key);
    abort();
  }
}


enkf_site_config_type * enkf_site_config_bootstrap(const char * _config_file) {
  const char * config_file = getenv("ENKF_SITE_CONFIG");
  if (config_file == NULL)
    config_file = _config_file;
  
  if (config_file == NULL) {
    fprintf(stderr,"%s: main enkf_config file is not set. Use environment variable \"ENKF_SITE_CONFIG\" - or recompile - aborting.\n",__func__);
    abort();
  }
  if (util_file_exists(config_file)) {
    enkf_site_config_type * site;
    site = util_malloc(sizeof * site , __func__);
    site->config = hash_alloc(10);
    enkf_site_config_add_node(site , "QUEUE_SYSTEM"  	  , NULL , 3 , (const char *[3]) {"RSH" , "LSF" , "LOCAL"} , enkf_site_config_validate_queue_system);
    enkf_site_config_add_node(site , "LSF_QUEUE"     	  , NULL , 0 , NULL , enkf_site_config_validate_queue_name);
    enkf_site_config_add_node(site , "LSF_RESOURCES" 	  , NULL , 0 , NULL , NULL);
    enkf_site_config_add_node(site , "START_ECLIPSE_CMD"  , NULL , 0 , NULL , enkf_site_config_assert_set_executable); 
    enkf_site_config_add_node(site , "ECLIPSE_EXECUTABLE" , NULL , 0 , NULL , enkf_site_config_assert_eclipse_executable); 
    enkf_site_config_add_node(site , "ECLIPSE_LD_PATH"    , NULL , 0 , NULL , NULL); 
    enkf_site_config_add_node(site , "LICENSE_SERVER"     , NULL , 0 , NULL , enkf_site_config_assert_set);
    enkf_site_config_add_node(site , "ECLIPSE_CONFIG"     , NULL , 0 , NULL , enkf_site_config_assert_set_existing);
    enkf_site_config_add_node(site , "MAX_RUNNING_LSF"    , NULL , 0 , NULL , NULL);
    enkf_site_config_add_node(site , "MAX_RUNNING_LOCAL"  , NULL , 0 , NULL , NULL);
    enkf_site_config_add_node(site , "MAX_RUNNING_RSH"    , NULL , 0 , NULL , NULL);
    enkf_site_config_add_node(site , "RSH_HOST_LIST"      , NULL , 0 , NULL , NULL);
    enkf_site_config_add_node(site , "RSH_COMMAND"        , NULL , 0 , NULL , NULL);

    {
      FILE * stream = util_fopen(config_file , "r");
      bool at_eof = false;
      char * key , * value;
      while (!at_eof) {
	key = util_fscanf_alloc_token(stream);
	if (key != NULL) {
	  value = util_fscanf_alloc_line(stream , &at_eof);
	  if (value != NULL) {
	    if (enkf_site_config_has_key(site , key))
	      enkf_site_config_set_key(site , key , value);
	    else
	      fprintf(stderr,"** Warning: key:%s is not recognized - line ignored \n",key);
	  free(value);
	  } 
	  free(key);
	} else 
	  util_forward_line(stream , &at_eof);
      }
      fclose(stream);
    }
    enkf_site_config_validate(site);
    return site;
  } else {
    fprintf(stderr,"%s: main config_file: %s not found - aborting \n",__func__ , config_file);
    abort();
  }
}



void enkf_site_config_validate(enkf_site_config_type *site) {
  bool valid_site_config = true;
  char ** key_list = hash_alloc_keylist(site->config);
  int ikey;

  for (ikey=0; ikey < hash_get_size(site->config); ikey++) {
    enkf_site_config_node_type * node = hash_get(site->config , key_list[ikey]);
    valid_site_config = (valid_site_config && enkf_site_config_node_validate(site , node));
  }
  hash_free_ext_keylist(site->config , key_list);
  if (!valid_site_config) {
    fprintf(stderr,"%s: configuration errors - aborting \n",__func__);
    abort();
  }
}



void  enkf_site_config_free(enkf_site_config_type * site) {
  hash_free(site->config);
  free(site);
}
