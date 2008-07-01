#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <config.h>
#include <ecl_util.h>
#include <gen_data_config.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <pthread.h>


/**
   Observe that this config object is prinsipally a bit different from
   the other _config objects, in that for the gen_data type more of
   the meta information is owned by the actual storage objects.
*/


struct gen_data_config_struct {
  CONFIG_STD_FIELDS;
  int      num_active;        /* The number of timesteps where this gen_data instance is active. */
  int     *active_reports;    /* A list of timesteps where this gen_data keyword is active. */
  char   **tag_list;          /* The remote tags we are looking for - can be NULL.*/
  char   **ecl_file_list;     /* The remote file we will load from. */


  /*-----------------------------------------------------------------*/
  /* Because the actual gen_data instances bootstrap from a file, the
     themselves contain meta information about size, ctype and so
     on. However these quantities must be equal for all gen_data
     instances (else all hell is loose ...), therefor gen_data
     instances call an xxx_assert routine on the gen_data_config
     object, and that routine checks that everything is equal. The
     variables below here are support variables for this
     functionality.
  */
  int               __report_step;   /* The current active report_step. */
  ecl_type_enum     __ecl_type;     
  char            * __file_tag;
  pthread_mutex_t   update_lock;
};



static gen_data_config_type * gen_data_config_alloc_empty( enkf_var_type var_type ) {
  gen_data_config_type * config = util_malloc(sizeof * config , __func__);
  config->data_size  	 = 0;
  config->var_type   	 = var_type;
  config->num_active 	 = 0;
  config->active_reports = NULL;
  config->tag_list       = NULL;
  config->ecl_file_list  = NULL;
  config->__file_tag     = NULL;
  config->__report_step  = -1;
  pthread_mutex_init( &config->update_lock , NULL );
  return config;
}



void gen_data_config_free(gen_data_config_type * config) {
  util_safe_free(config->active_reports);
  util_safe_free(config->__file_tag);
  util_free_stringlist(config->tag_list      , config->num_active);
  util_free_stringlist(config->ecl_file_list , config->num_active);
  free(config);
}


/**
   This function gets metadata (from a gen_data instance). Iff the
   stored variable __report_step is equal to the input report_step, it
   verifies that the stored values of size, ecl_type and file_tag also
   correspond to the input values - if this is not the case it will
   ABORT().

   If the stored report_step deviates from the input report_step, it
   will just store the new metadata.
*/

void gen_data_config_assert_metadata(gen_data_config_type * config , int report_step , int size , ecl_type_enum ecl_type, const char * file_tag) {
  pthread_mutex_lock( &config->update_lock );
  {
    if (report_step == config->__report_step) {
      if (config->data_size   != size)     		util_abort("%s: tried to combine gen_data instances of different size.\n",__func__);
      if (config->__ecl_type  != ecl_type) 		util_abort("%s: tried to combine gen_data instances with different type.\n",__func__);
      if (strcmp(file_tag , config->__file_tag) != 0)   util_abort("%s: tried to combine gen_data instances with different file_tag.\n." , __func__);
    } else {
      config->__report_step = report_step;
      config->data_size     = size;
      config->__ecl_type    = ecl_type;
      config->__file_tag    = util_realloc_string_copy(config->__file_tag , file_tag);
    }
  }
  pthread_mutex_unlock( &config->update_lock );
}


/** 
    Determines if the gen_data_config instance is active at this
    report_step.
*/

bool gen_data_config_is_active(const gen_data_config_type * config , int report_step) {
  int istep;
  bool active = false;
  
  for (istep = 0; istep < config->num_active; istep++)
    if (report_step == config->active_reports[istep])
      active = true;
  
  return active;
}



static void gen_data_config_add_active(gen_data_config_type * gen_config , const char * ecl_file , int report_step , const char * tag) {
  gen_config->num_active++;
  gen_config->active_reports = util_realloc(gen_config->active_reports , gen_config->num_active * sizeof * gen_config->active_reports , __func__);
  gen_config->ecl_file_list  = util_realloc(gen_config->ecl_file_list  , gen_config->num_active * sizeof * gen_config->ecl_file_list  , __func__);
  gen_config->tag_list       = util_realloc(gen_config->tag_list       , gen_config->num_active * sizeof * gen_config->tag_list       , __func__);

  gen_config->active_reports[gen_config->num_active - 1] = report_step;
  gen_config->ecl_file_list[gen_config->num_active - 1]  = util_alloc_string_copy(ecl_file);
  gen_config->tag_list[gen_config->num_active - 1]       = util_alloc_string_copy(tag);
}


/**
   This function returns, by reference, the eclips file to load from,
   and the tag to ask for, at a particular report_step. Observe that
   it is assumed that the function gen_data_config_is_active() has
   already been queried to determine whether the report_step is indeed
   active; the function will fail hard if the report step is not active.
*/

void gen_data_config_get_ecl_file(const gen_data_config_type * config , int report_step , char ** ecl_file , char ** file_tag) {
  bool found = false;
  int istep;
  
  for (istep = 0; istep < config->num_active; istep++)
    if (report_step == config->active_reports[istep]) {
      found = true;
      *ecl_file = config->ecl_file_list[istep];
      *file_tag = config->tag_list[istep];
    }

  if (!found) 
    util_abort("%s: asked for ecl_file / file_tag in at report_step:%d - this report step is not active\n",__func__ , report_step);
}




/**
   This function bootstraps a gen_data_config object from a configuration
   file. The format of the configuration file is as follows:

   ECL_FILE1   REPORT_STEP1   <TAG1>
   ECL_FILE2   REPORT_STEP2   <TAG2>
   ....

   Here ECL_FILE is the name of a file which is loaded, report_step is
   the report_step where we load this file, and tag is a tag we expect
   to find in the file. The tag is optional, and can be left blank.
*/

gen_data_config_type * gen_data_config_fscanf_alloc(const char * config_file) {
  gen_data_config_type * gen_config = gen_data_config_alloc_empty( ecl_restart );
  config_type * config = config_alloc(true);
  config_parse(config , config_file , "--");
  {
    int    iarg , argc;
    int    num_active;
    char **ecl_file_list;
    
    ecl_file_list = config_alloc_active_list(config , &num_active);
    for (iarg = 0; iarg < num_active; iarg++) {
      const char  * ecl_file = ecl_file_list[iarg];
      const char ** argv     = config_get_argv(config , ecl_file , &argc);
      const char  * tag;
      int report_step;
      
      if (argc == 0) 
	util_exit("%s: missing report step when parsing:%s in %s \n",__func__ , ecl_file , config_file);

      if (!util_sscanf_int(argv[0] , &report_step)) 
	util_exit("%s: failed to parse out report_step as integer - aborting \n",__func__);

      if (argc == 1)
	tag = NULL;
      else
	tag = argv[1];
      
      gen_data_config_add_active(gen_config , ecl_file , report_step , tag);
    }
    util_free_stringlist(ecl_file_list , num_active);
  }
  config_free(config);
  return gen_config;
}
