#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <config.h>
#include <ecl_util.h>
#include <gen_data_config.h>
#include <gen_common.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <pthread.h>


/**
   Observe that this config object is principally quite different from
   the other _config objects for two reasons:

   1. The gen_data type contains more of the meta information in the
      actual storage objects.

   2. The config object serves as config object for *both* the data (gen_data),
      and for the observations (gen_obs).

*/


struct gen_data_config_struct {
  CONFIG_STD_FIELDS;
  int      num_active;        /* The number of timesteps where this gen_data instance is active. */
  int     *active_reports;    /* A list of timesteps where this gen_data keyword is active. */
  char   **config_tag_list;   /* The remote tags we are looking for - can be NULL.*/
  char   **ecl_file_list;     /* The remote file we will load forecasts from.*/
  char   **obs_file_list;     /* A list of files we load observations from ?? */
  bool    *obs_active;        /* Whether the observation is active (for this report step) - i.e. should be used in the EnKF update step. */   


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
  int               active_size;
  int              *active_list;
  int               __report_step;   /* The current active report_step. */
  int               __report_index; 
  int               __obs_size;      /* Storing the size of (currently active) observation, and comparing with gen_data measurements MUST agree. */
  ecl_type_enum     __ecl_type;     
  char            * __file_tag;      /* Observe that this is a tag found in a gen_data file. The gen_data object
					has already asserted that it matches the config_tag. */     
  pthread_mutex_t   update_lock;
};


/**
   Takes a report_step as input, and returns the index (in the
   gen_data_config) object of this report_step, or -1 if the
   report_step is not active.
*/

static int gen_data_config_get_report_index(const gen_data_config_type * config , int report_step) {
  int istep;
  int report_index = -1;
  
  for (istep = 0; istep < config->num_active; istep++)
    if (report_step == config->active_reports[istep])
      report_index = istep;
  
  return report_index;
}


static gen_data_config_type * gen_data_config_alloc_empty( enkf_var_type var_type ) {
  gen_data_config_type * config = util_malloc(sizeof * config , __func__);
  config->data_size  	 = 0;
  config->var_type   	 = var_type;
  config->num_active 	 = 0;
  config->active_reports = NULL;
  config->config_tag_list       = NULL;
  config->ecl_file_list  = NULL;
  config->obs_file_list  = NULL;
  config->obs_active     = NULL;    
  config->active_list    = NULL;
  config->__file_tag     = NULL;
  config->__report_step  = -1;
  pthread_mutex_init( &config->update_lock , NULL );
  {
    config->__report_index = 0;
    gen_data_config_deactivate_metadata(config);
  }
  return config;
}


int gen_data_config_get_data_size(const gen_data_config_type * config) { return config->data_size; }


void gen_data_config_free(gen_data_config_type * config) {
  util_safe_free(config->active_reports);
  util_safe_free(config->__file_tag);
  util_free_stringlist(config->config_tag_list      , config->num_active);
  util_free_stringlist(config->ecl_file_list , config->num_active);
  util_free_stringlist(config->obs_file_list , config->num_active);
  util_safe_free(config->obs_active);
  util_safe_free(config->active_list);
  free(config);
}



/* 
   This function returns (by reference) the name of the observation file, and
   the config tag currently active. This function does not accept a report_step,
   so we start by checking that __report_index is valid, if not we die().
*/


/** 
    This function loads the active map from the obs_file: Observe that
    it is assumed that the header, and the data+std of the observation
    has already been read in from the open stream.
*/

void gen_data_config_fload_iactive(gen_data_config_type * config ,  FILE * stream , const char * obs_file ,gen_data_file_type  file_type, int size) {
  int i;
  int active_size = 0;
  int * active_list = util_malloc(size * sizeof * active_list , __func__);
  int * iactive_int = util_malloc(size * sizeof * iactive_int , __func__);
  gen_common_fload_data( stream , obs_file , file_type , ecl_int_type , size , iactive_int );
  
  for (i = 0; i < size; i++)
    if (iactive_int[i] == 1) {
      active_list[active_size] = i;
      active_size++;
    } else {
      if (iactive_int[i] != 0) {
	fprintf(stderr,"Observation file:%s \n",obs_file);
	util_abort("%s: the active map in the observation file must have ONLY 0:not active and 1:active. %d invalid \n",__func__, iactive_int[i]);
      }
    }

  free(iactive_int);
  util_safe_free(config->active_list);
  config->active_list = util_alloc_copy(active_list , active_size * sizeof * active_list , __func__);
  free(active_list);
  config->active_size = active_size;
}


/**
   This function gets metadata (from a gen_data instance). Iff the
   stored variable __report_step is equal to the input report_step, it
   verifies that the stored values of size, ecl_type and config_tag also
   correspond to the input values - if this is not the case it will
   ABORT().

   If the stored report_step deviates from the input report_step, it
   will just store the new metadata.
*/

void gen_data_config_assert_metadata(gen_data_config_type * config , int report_step , int size , ecl_type_enum ecl_type, const char * file_tag) {
  pthread_mutex_lock( &config->update_lock );
  {
    if (report_step == config->__report_step) {
      if (config->data_size   != size)     		    util_abort("%s: tried to combine gen_data instances of different size.\n",__func__);
      if (config->__ecl_type  != ecl_type) 		    util_abort("%s: tried to combine gen_data instances with different type.\n",__func__);
      if (strcmp(file_tag , config->__file_tag) != 0)       util_abort("%s: tried to combine gen_data instances with different config_tag.\n." , __func__);
      if (config->__obs_size >= 0) 
	if (config->__obs_size != size) util_abort("%s: size mismatch: Observation:%d   forward_model:%d \n",__func__ , config->__obs_size , size);

    } else {
      config->__report_index = gen_data_config_get_report_index(config , report_step);
      if (config->__report_index < 0) 
	util_abort("%s internal error - called at inactive report step \n",__func__);
      {
	config->__report_step  = report_step;
	config->data_size      = size;
	config->__ecl_type     = ecl_type;
	config->__file_tag     = util_realloc_string_copy(config->__file_tag , file_tag);

	
      }
    }
  }
  pthread_mutex_unlock( &config->update_lock );
}







/** This function does the opposite of gen_data_config_assert_metadata(). When a
    gen_data instance has been deactivated, it calls this function, asserting
    the config object agrees that no data is currently active.
*/
    
void gen_data_config_deactivate_metadata(gen_data_config_type * config) {
  pthread_mutex_lock( &config->update_lock );
  {
    if (config->__report_index >= 0) {
      config->active_list    = util_safe_free(config->active_list);
      config->__file_tag     = util_safe_free(config->__file_tag); 
      config->__report_index = -1;
      config->__report_step  = -1;
      config->data_size      =  0;
      config->active_size    =  0;
      config->__obs_size     =  -1; /* Invalid */
    }
  }
  pthread_mutex_unlock( &config->update_lock ); 
}



/** 
    Determines if the gen_data_config instance is active at this
    report_step.
*/

bool gen_data_config_is_active(const gen_data_config_type * config , int report_step) {
  if (gen_data_config_get_report_index(config , report_step) >= 0)
    return true;
  else
    return false;
}



static void gen_data_config_add_active(gen_data_config_type * gen_config , const char * obs_file , const char * ecl_file , int report_step , bool obs_active , const char * tag) {
  if (gen_data_config_get_report_index(gen_config , report_step) >= 0)
    util_abort("%s: Report_step must be unique (at least for now...) \n",__func__);
  {
    gen_config->num_active++;
    gen_config->active_reports  = util_realloc(gen_config->active_reports  , gen_config->num_active * sizeof * gen_config->active_reports , __func__);
    gen_config->ecl_file_list   = util_realloc(gen_config->ecl_file_list   , gen_config->num_active * sizeof * gen_config->ecl_file_list  , __func__);
    gen_config->obs_file_list   = util_realloc(gen_config->obs_file_list   , gen_config->num_active * sizeof * gen_config->obs_file_list  , __func__);
    gen_config->config_tag_list = util_realloc(gen_config->config_tag_list , gen_config->num_active * sizeof * gen_config->config_tag_list       , __func__);
    gen_config->obs_active      = util_realloc(gen_config->obs_active      , gen_config->num_active * sizeof * gen_config->obs_active     , __func__);
    
    {
      int report_index = gen_config->num_active - 1;
      gen_config->active_reports[report_index]  = report_step;
      gen_config->ecl_file_list[report_index]   = util_alloc_string_copy(ecl_file);
      gen_config->obs_file_list[report_index]   = util_alloc_string_copy(obs_file);
      gen_config->config_tag_list[report_index] = util_alloc_string_copy(tag);
      gen_config->obs_active[report_index]      = obs_active;
    }
  }
}




/*
  This function check if the global ON|OFF flag is on or off. For the
  currently loaded report_step. It is called from gen_obs(), which
  does not know thereport_number - i.e. we must assume that the
  report_index currently has a valid value, ant abort otherwise.
*/

bool gen_data_config_obs_on(const gen_data_config_type * config, int report_step) {
  int report_index = gen_data_config_get_report_index(config , report_step);
  if (report_index < 0)
    return false;
  else {
    return config->obs_active[report_index];
  }
}




/**
   This function returns, by reference, the eclipse file to load from,
   and the tag to ask for, at a particular report_step. Observe that
   it is assumed that the function gen_data_config_is_active() has
   already been queried to determine whether the report_step is indeed
   active; the function will fail hard if the report step is not active.
*/

void gen_data_config_get_ecl_file(const gen_data_config_type * config , int report_step , char ** ecl_file , char ** config_tag) {
  bool found = false;
  int istep;
  
  for (istep = 0; istep < config->num_active; istep++)
    if (report_step == config->active_reports[istep]) {
      found = true;
      *ecl_file = config->ecl_file_list[istep];
      *config_tag = config->config_tag_list[istep];
    }

  if (!found) 
    util_abort("%s: asked for ecl_file / config_tag in at report_step:%d - this report step is not active\n",__func__ , report_step);
}





void gen_data_config_get_obs_file(const gen_data_config_type * config , int report_step , char ** obs_file , char ** config_tag) {
  int report_index = gen_data_config_get_report_index(config , report_step);
  if (report_index >= 0) {
    *obs_file   = config->obs_file_list[report_index];
    *config_tag = config->config_tag_list[report_index];
  } else {
    *obs_file   = NULL ; 
    *config_tag = NULL ; 
  }
}


/**
   The __obs_size field is used to check that the size of the
   observation vector, and the data are of the same size. 
*/

void gen_data_config_set_obs_size(gen_data_config_type * config, int obs_size) { config->__obs_size = obs_size; }


/**
   This function bootstraps a gen_data_config object from a configuration
   file. The format of the configuration file is as follows:

   OBS_FILE1  ECL_FILE1   REPORT_STEP1   ON|OFF   <TAG1>  
   OBS_FILE2  ECL_FILE2   REPORT_STEP2   ON|OFF   <TAG2>
   ....

   Here ECL_FILE is the name of a file which is loaded, report_step is
   the report_step where we load this file, and tag is a tag we expect
   to find in the file. The tag is optional, and can be left
   blank. The report_steps must (currently) be unique.
*/

gen_data_config_type * gen_data_config_fscanf_alloc(const char * config_file) {
  gen_data_config_type * gen_config = gen_data_config_alloc_empty( dynamic );
  config_type * config = config_alloc( );
  config_parse(config , config_file , "--" , NULL , true , true);
  {
    int    iarg;
    int    num_active;
    char **obs_file_list;

    obs_file_list = config_alloc_active_list(config , &num_active);
    
    for (iarg = 0; iarg < num_active; iarg++) {
      const char  * obs_file = obs_file_list[iarg];
      const stringlist_type * argv = config_get_stringlist_ref(config , obs_file);
      const char  * ecl_file;
      const char  * tag;
      bool  obs_active = false; /* Dummy init */
      int report_step;
      int argc = stringlist_get_size(argv);
  
      if (argc <= 2) 
	util_exit("%s: missing report step / ON|OFF when parsing:%s in %s \n",__func__ , obs_file , config_file);
      
      ecl_file = stringlist_iget(argv , 0);
      if (!util_sscanf_int(stringlist_iget(argv , 1) , &report_step)) 
	util_exit("%s: failed to parse out report_step as integer - aborting \n",__func__);
      
      {
	char * on_off = util_alloc_string_copy( stringlist_iget(argv , 2) );
	util_strupr( on_off );
	if (strcmp(on_off , "ON") == 0)
	  obs_active = true;
	else if (strcmp(on_off , "OFF") == 0)
	  obs_active = false;
	else 
	  util_abort("%s: item nr four must be ON or OFF. \n",__func__);
	free( on_off );
      }
      
      if (argc == 3)
	tag = NULL;
      else
	tag = stringlist_iget(argv , 3);
      
      gen_data_config_add_active(gen_config , obs_file , ecl_file , report_step , obs_active , tag);
    }
    util_free_stringlist(obs_file_list , num_active);
  }
  config_free(config);
  return gen_config;
}


/*****************************************************************/

VOID_FREE(gen_data_config)
GET_ACTIVE_SIZE(gen_data)
GET_ACTIVE_LIST(gen_data)
