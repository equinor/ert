#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <util.h>
#include <stdlib.h>
#include <string.h>
#include <path_fmt.h>
#include <enkf_sched.h>
#include <model_config.h>
#include <hash.h>
#include <history.h>
#include <config.h>
#include <sched_file.h>
#include <ecl_sum.h>
#include <ecl_util.h>
#include <ecl_grid.h>
#include <menu.h>
#include <enkf_types.h>
#include <plain_driver.h>
#include <forward_model.h>
#include <bool_vector.h>
#include <fs_types.h>
#include <enkf_defaults.h>

/**
   This struct contains configuration which is specific to this
   particular model/run. Such of the information is actually accessed
   directly through the enkf_state object; but this struct is the
   owner of the information, and responsible for allocating/freeing
   it.

   Observe that the distinction of what goes in model_config, and what
   goes in ecl_config is not entirely clear; ECLIPSE is unfortunately
   not (yet ??) exactly 'any' reservoir simulator in this context.

   Read the documentation about restart numbering in enkf_sched.c
*/


struct model_config_struct {
  stringlist_type     * case_names;                 /* A list of "iens -> name" mappings - can be NULL. */
  forward_model_type  * std_forward_model;   	    /* The forward_model - as loaded from the config file. Each enkf_state object internalizes its private copy of the forward_model. */  
  bool                  use_lsf;             	    /* The forward models need to know whether we are using lsf. */  
  history_type        * history;             	    /* The history object. */
  path_fmt_type       * runpath;             	    /* path_fmt instance for runpath - runtime the call gets arguments: (iens, report_step1 , report_step2) - i.e. at least one %d must be present.*/  
  enkf_sched_type     * enkf_sched;          	    /* The enkf_sched object controlling when the enkf is ON|OFF, strides in report steps and special forward model - allocated on demand - right before use. */ 
  char                * enkf_sched_file;     	    /* THe name of file containg enkf schedule information - can be NULL to get default behaviour. */
  char                * enspath;
  fs_driver_impl        dbase_type;
  int                   last_history_restart;       /* The end of the history - this is inclusive.*/
  bool                  resample_when_fail;         /* Should we resample when a model fails to integrate? */
  bool                  has_prediction; 
  int                   max_internal_submit;        /* How many times to retry if the load fails. */
  bool_vector_type    * internalize_state;   	    /* Should the (full) state be internalized (at this report_step). */
  bool_vector_type    * internalize_results; 	    /* Should the results (i.e. summary in ECLIPSE speak) be internalized at this report_step? */
  bool_vector_type    * __load_state;        	    /* Internal variable: is it necessary to load the state? */
  bool_vector_type    * __load_results;      	    /* Internal variable: is it necessary to load the results? */
};





void model_config_set_runpath_fmt(model_config_type * model_config, const char * fmt){
  if (model_config->runpath != NULL)
    path_fmt_free( model_config->runpath );
  
  model_config->runpath  = path_fmt_alloc_directory_fmt( fmt );
}

/**
   This function is not called at bootstrap time, but rather as part of an initialization
   just before the run. Can be called maaaanye times for one application invokation.

   Observe that the 'total' length is set as as the return value from this function.
*/


void model_config_set_enkf_sched(model_config_type * model_config , const ext_joblist_type * joblist , run_mode_type run_mode , bool statoil_mode) {
  if (model_config->enkf_sched != NULL)
    enkf_sched_free( model_config->enkf_sched );
  
  model_config->enkf_sched  = enkf_sched_fscanf_alloc(model_config->enkf_sched_file       , 
						      model_config->last_history_restart  , 
						      run_mode                            , 
						      joblist                             , 
						      statoil_mode , 
						      model_config->use_lsf );
}


void model_config_set_enkf_sched_file(model_config_type * model_config , const char * enkf_sched_file) {
  model_config->enkf_sched_file = util_realloc_string_copy( model_config->enkf_sched_file , enkf_sched_file);
}


void model_config_set_enspath( model_config_type * model_config , const char * enspath) {
  model_config->enspath = util_realloc_string_copy( model_config->enspath , enspath );
}


void model_config_set_dbase_type( model_config_type * model_config , const char * dbase_type_string) {
  model_config->dbase_type = fs_types_lookup_string_name( dbase_type_string );
  if (model_config->dbase_type == INVALID_DRIVER_ID)
    util_abort("%s: did not recognize driver_type:%s \n",__func__ , dbase_type_string);
}


const char * model_config_get_enspath( const model_config_type * model_config) {
  return model_config->enspath;
}


fs_driver_impl model_config_get_dbase_type(const model_config_type * model_config ) {
  return model_config->dbase_type;
}


model_config_type * model_config_alloc(const config_type * config , int ens_size , const ext_joblist_type * joblist , int last_history_restart , const sched_file_type * sched_file , bool statoil_mode , bool use_lsf) {
  model_config_type * model_config        = util_malloc(sizeof * model_config , __func__);
  /**
     There are essentially three levels of initialisation:

     1. Initialize to NULL / invalid.
     2. Initialize with default values.
     3. Initialize with user supplied values.

  */
  model_config->case_names                = NULL;
  model_config->use_lsf                   = use_lsf;
  model_config->max_internal_submit       = DEFAULT_MAX_INTERNAL_SUBMIT;
  model_config->resample_when_fail        = DEFAULT_RESAMPLE_WHEN_FAIL;
  model_config->enspath                   = NULL;
  model_config->dbase_type                = INVALID_DRIVER_ID;
  model_config->runpath                   = NULL;
  model_config->enkf_sched                = NULL;
  model_config->enkf_sched_file           = NULL;   
  model_config->std_forward_model         = forward_model_alloc(  joblist , statoil_mode , model_config->use_lsf , DEFAULT_START_TAG , DEFAULT_END_TAG );
  
  {
    char * config_string = config_alloc_joined_string( config , "FORWARD_MODEL" , " ");
    forward_model_parse_init( model_config->std_forward_model , config_string );
    free(config_string);
  }
  
  model_config_set_enspath( model_config , DEFAULT_ENSPATH );
  model_config_set_dbase_type( model_config , DEFAULT_DBASE_TYPE );
  model_config_set_runpath_fmt( model_config , DEFAULT_RUNPATH);


  if (config_item_set( config , "ENKF_SCHED_FILE"))
    model_config_set_enkf_sched_file(model_config , config_get_value(config , "ENKF_SCHED_FILE" ));
  
  if (config_item_set( config, "RUNPATH"))
    model_config_set_runpath_fmt( model_config , config_get_value(config , "RUNPATH") );

  {
    model_config->history              = history_alloc_from_sched_file(sched_file);  
    model_config->last_history_restart = last_history_restart;
  }
  
  {
    const char * history_source = config_iget(config , "HISTORY_SOURCE", 0,0);
    const char * refcase        = NULL;
    bool  use_history;

    if (strcmp(history_source , "REFCASE_SIMULATED") == 0) {
      refcase = config_iget(config , "REFCASE" , 0,0);
      use_history = false;
    } else if (strcmp(history_source , "REFCASE_HISTORY") == 0) {
      refcase = config_iget(config , "REFCASE" , 0,0);
      use_history = true;
    }

    if ((refcase != NULL) && (strcmp(history_source , "SCHEDULE") != 0)) {
      printf("Loading summary from: %s \n",refcase);
      history_realloc_from_summary( model_config->history , refcase , use_history);        
    }
  }

  {
    int num_restart = history_get_num_restarts(model_config->history);
    model_config->internalize_state   = bool_vector_alloc( num_restart , false );
    model_config->internalize_results = bool_vector_alloc( num_restart , false );
    model_config->__load_state        = bool_vector_alloc( num_restart , false ); 
    model_config->__load_results      = bool_vector_alloc( num_restart , false );
  }

  /*
    The full treatment of the SCHEDULE_PREDICTION_FILE keyword is in
    the ensemble_config file, because the functionality is implemented
    as (quite) plain GEN_KW instance. Here we just check if it is
    present or not.
  */
  
  if (config_item_set(config ,  "SCHEDULE_PREDICTION_FILE")) 
    model_config->has_prediction = true;
  else
    model_config->has_prediction = false;


  if (config_item_set(config ,  "CASE_TABLE")) {
    bool atEOF = false;
    char casename[128];
    int  case_size = 0;
    FILE * stream = util_fopen( config_iget( config , "CASE_TABLE" , 0,0) , "r");
    model_config->case_names = stringlist_alloc_new();
    while (!atEOF) {
      if (fscanf( stream , "%s" , casename) == 1) {
        stringlist_append_copy( model_config->case_names , casename );
        case_size++;
      } else
        atEOF = true;
    }
    fclose( stream );

    if (case_size < ens_size) {
      for (int i = case_size; i < ens_size; i++)
        stringlist_append_owned_ref( model_config->case_names , util_alloc_sprintf("case_%04d" , i));
      fprintf(stderr, "** Warning: mismatch between NUM_REALIZATIONS:%d and size of CASE_TABLE:%d - using \'case_nnnn\' for the last cases %d.\n", ens_size , case_size , ens_size - case_size);
    } else if (case_size > ens_size) 
      fprintf(stderr, "** Warning: mismatch between NUM_REALIZATIONS:%d and CASE_TABLE:%d - only the %d realizations will be used.\n", ens_size , case_size , ens_size);
  }
    
  
  if (config_item_set( config , "ENSPATH"))
    model_config_set_enspath( model_config , config_get_value(config , "ENSPATH"));

  if (config_item_set( config , "DBASE_TYPE"))
    model_config_set_dbase_type( model_config , config_get_value(config , "DBASE_TYPE"));
  
  if (config_item_set( config , "MAX_RETRY"))
    model_config->max_internal_submit = config_iget_as_int( config , "MAX_RETRY" , 0,0);

  if (config_item_set( config , "RESAMPLE_WHEN_FAIL"))
    model_config->resample_when_fail        = config_iget_as_bool( config , "RESAMPLE_WHEN_FAIL" ,0,0);

  return model_config;
}


const char * model_config_iget_casename( const model_config_type * model_config , int index) {
  if (model_config->case_names == NULL)
    return NULL;
  else
    return stringlist_iget( model_config->case_names , index );
}



void model_config_free(model_config_type * model_config) {
  path_fmt_free(  model_config->runpath );
  if (model_config->enkf_sched != NULL)
    enkf_sched_free( model_config->enkf_sched );
  free( model_config->enspath );
  util_safe_free( model_config->enkf_sched_file );
  history_free(model_config->history);
  forward_model_free(model_config->std_forward_model);
  bool_vector_free(model_config->internalize_results);
  bool_vector_free(model_config->internalize_state);
  bool_vector_free(model_config->__load_state);
  bool_vector_free(model_config->__load_results);
  if (model_config->case_names != NULL) stringlist_free( model_config->case_names );
  free(model_config);
}



path_fmt_type * model_config_get_runpath_fmt(const model_config_type * model_config) {
  return model_config->runpath;
}


enkf_sched_type * model_config_get_enkf_sched(const model_config_type * config) {
  return config->enkf_sched;
}

history_type * model_config_get_history(const model_config_type * config) {
  return config->history;
}


/**
   Because the different enkf_state instances can have different
   schedule prediction files they can in principle have different
   number of dates. This variable only records the longest. Observe
   that the input is assumed to be transformed to the "Number of last
   restart file" domain. I.e. in the case of four restart files:
   
      num_restart_files = 4 => "0000","0001","0002","0003"

   This function expects to get the input three.           

*/



int model_config_get_last_history_restart(const model_config_type * config) {
  return config->last_history_restart;
}


bool model_config_has_prediction(const model_config_type * config) {
  return config->has_prediction;
}


void model_config_interactive_set_runpath__(void * arg) {
  arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
  model_config_type * model_config = arg_pack_iget_ptr(arg_pack , 0);
  menu_item_type    * item         = arg_pack_iget_ptr(arg_pack , 1);
  char runpath_fmt[256];
  printf("Give runpath format ==> ");
  scanf("%s" , runpath_fmt);
  model_config_set_runpath_fmt(model_config , runpath_fmt);
  {
    char * menu_label = util_alloc_sprintf("Set new value for RUNPATH:%s" , runpath_fmt);
    menu_item_set_label( item , menu_label );
    free(menu_label);
  }
}


forward_model_type * model_config_get_std_forward_model( const model_config_type * config) {
  return config->std_forward_model;
}


/*****************************************************************/

/* Setting everything back to the default value: false. */
void model_config_init_internalization( model_config_type * config ) {
  bool_vector_reset(config->internalize_state);
  bool_vector_reset(config->__load_state);
  bool_vector_reset(config->internalize_results);
  bool_vector_reset(config->__load_results);
}


/**
   This function sets the internalize_state flag to true for
   report_step. Because of the coupling to the __load_state variable
   this function can __ONLY__ be used to set internalize to true. 
*/

void model_config_set_internalize_state( model_config_type * config , int report_step) {
  bool_vector_iset(config->internalize_state , report_step , true);
  bool_vector_iset(config->__load_state      , report_step , true);
}


void model_config_set_internalize_results( model_config_type * config , int report_step) {
  bool_vector_iset(config->internalize_results , report_step , true);
  bool_vector_iset(config->__load_results      , report_step , true);
}

void model_config_set_load_results( model_config_type * config , int report_step) {
  bool_vector_iset(config->__load_results , report_step , true);
}

void model_config_set_load_state( model_config_type * config , int report_step) {
  bool_vector_iset(config->__load_state , report_step , true);
}



/* Query functions. */

bool model_config_internalize_state( const model_config_type * config , int report_step) {
  return bool_vector_iget(config->internalize_state , report_step);
}

bool model_config_internalize_results( const model_config_type * config , int report_step) {
  return bool_vector_iget(config->internalize_results , report_step);
}

/*****************************************************************/

bool model_config_load_state( const model_config_type * config , int report_step) {
  return bool_vector_iget(config->__load_state , report_step);
}

bool model_config_load_results( const model_config_type * config , int report_step) {
  return bool_vector_iget(config->__load_results , report_step);
}

bool model_config_resample_when_fail( const model_config_type * config ) {
  return config->resample_when_fail;
}

int model_config_get_max_internal_submit( const model_config_type * config ) {
  return config->max_internal_submit;
}
