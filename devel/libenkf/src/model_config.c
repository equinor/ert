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
#include "config_keys.h"

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
  stringlist_type      * case_names;                 /* A list of "iens -> name" mappings - can be NULL. */
  char                 * case_table_file; 
  forward_model_type   * forward_model;             /* The forward_model - as loaded from the config file. Each enkf_state object internalizes its private copy of the forward_model. */  
  history_type         * history;                   /* The history object. */
  path_fmt_type        * runpath;                   /* path_fmt instance for runpath - runtime the call gets arguments: (iens, report_step1 , report_step2) - i.e. at least one %d must be present.*/  
  enkf_sched_type      * enkf_sched;                /* The enkf_sched object controlling when the enkf is ON|OFF, strides in report steps and special forward model - allocated on demand - right before use. */ 
  char                 * enkf_sched_file;           /* THe name of file containg enkf schedule information - can be NULL to get default behaviour. */
  char                 * enspath;
  char                 * select_case;
  fs_driver_impl         dbase_type;
  int                    last_history_restart;       /* The end of the history - this is inclusive.*/
  bool                   has_prediction; 
  int                    max_internal_submit;        /* How many times to retry if the load fails. */
  history_source_type    history_source;
  bool                   ignore_schedule; 
  const ecl_sum_type   * refcase;                    /* A pointer to the refcase - can be NULL. Observe that this ONLY a pointer 
                                                        to the ecl_sum instance owned and held by the ecl_config object. */
  /** The results are always loaded. */
  bool_vector_type    * internalize_state;          /* Should the (full) state be internalized (at this report_step). */
  bool_vector_type    * __load_state;               /* Internal variable: is it necessary to load the state? */
};




 path_fmt_type * model_config_get_runpath_fmt(const model_config_type * model_config) {
   return model_config->runpath;
 }

 const char * model_config_get_runpath_as_char( const model_config_type * model_config ) {
   return path_fmt_get_fmt( model_config->runpath );
 }


const char * model_config_get_case_table_file( const model_config_type * model_config ) {
  return model_config->case_table_file;
}

void model_config_set_case_table( model_config_type * model_config , int ens_size , const char * case_table_file ) {
  if (model_config->case_table_file != NULL) { /* Clear the current selection */
    free( model_config->case_table_file );
    stringlist_free( model_config->case_names );
    
    model_config->case_table_file = NULL;
    model_config->case_names      = NULL;
  }

  if (case_table_file != NULL) {
    bool atEOF = false;
    char casename[128];
    int  case_size = 0;
    FILE * stream = util_fopen( case_table_file , "r");
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
}


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


 void model_config_set_enkf_sched(model_config_type * model_config , const ext_joblist_type * joblist , run_mode_type run_mode ) {
   if (model_config->enkf_sched != NULL)
     enkf_sched_free( model_config->enkf_sched );

   model_config->enkf_sched  = enkf_sched_fscanf_alloc(model_config->enkf_sched_file       , 
                                                       model_config->last_history_restart  , 
                                                       run_mode);
                                                       
                                                       
 }


 void model_config_set_enkf_sched_file(model_config_type * model_config , const char * enkf_sched_file) {
   model_config->enkf_sched_file = util_realloc_string_copy( model_config->enkf_sched_file , enkf_sched_file);
 }

 char * model_config_get_enkf_sched_file(const model_config_type * model_config ) {
   return model_config->enkf_sched_file;
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



 void model_config_set_max_resample( model_config_type * model_config , int max_resample ) {
   model_config->max_internal_submit = max_resample;
 }


 int model_config_get_max_resample(const model_config_type * model_config ) {
   return model_config->max_internal_submit;
 }


void model_config_set_refcase( model_config_type * model_config , const ecl_sum_type * refcase ) {
  model_config->refcase = refcase;
}


history_source_type model_config_get_history_source( const model_config_type * model_config ) {
  return model_config->history_source;
}

/**
   Current implementation is ONLY prepared to do a reload from a
   schedule based history to a refcase based history.
*/

void model_config_set_history_source( model_config_type * model_config , history_source_type history_source ) {
  if (model_config->history_source != history_source) {
    bool use_history = false;

    model_config->history_source = history_source;
    if (history_source == REFCASE_HISTORY)
      use_history = true;
    if ((history_source != SCHEDULE) && (model_config->refcase == NULL))
      util_abort("%s: when using a REFCASE based history you must have a refcase installed. \n",__func__);
    
    if (history_source != SCHEDULE) {
      /* We want to use the REFCASE */
      
      if (model_config->refcase != NULL) 
        history_use_summary( model_config->history , model_config->refcase , use_history);        
      else
        util_exit("%s: Invalid configuration. When using HISTORY_SOURCE != SCHEDULE you must supply a REFCASE.\n",__func__);
    }
  }
}


model_config_type * model_config_alloc_empty() {
  model_config_type * model_config  = util_malloc(sizeof * model_config , __func__);
  /**
     There are essentially three levels of initialisation:

     1. Initialize to NULL / invalid.
     2. Initialize with default values.
     3. Initialize with user supplied values.

  */
  model_config->case_names                = NULL;
  model_config->max_internal_submit       = DEFAULT_MAX_INTERNAL_SUBMIT;
  model_config->enspath                   = NULL;
  model_config->dbase_type                = INVALID_DRIVER_ID;
  model_config->runpath                   = NULL;
  model_config->enkf_sched                = NULL;
  model_config->enkf_sched_file           = NULL;   
  model_config->case_table_file           = NULL;
  model_config->select_case               = NULL;    
  model_config->history                   = NULL;
  model_config->last_history_restart      = 0;
  model_config->internalize_state         = bool_vector_alloc( 0 , false );
  model_config->__load_state              = bool_vector_alloc( 0 , false ); 
  model_config->ignore_schedule           = DEFAULT_IGNORE_SCHEDULE;

  model_config_set_history_source( model_config , DEFAULT_HISTORY_SOURCE );
  model_config_set_enspath( model_config        , DEFAULT_ENSPATH );
  model_config_set_dbase_type( model_config     , DEFAULT_DBASE_TYPE );
  model_config_set_runpath_fmt( model_config    , DEFAULT_RUNPATH);
  
  return model_config;
}



void model_config_init(model_config_type * model_config , 
                       const config_type * config , 
                       int ens_size , 
                       const ext_joblist_type * joblist , 
                       int last_history_restart , 
                       const sched_file_type * sched_file , 
                       const ecl_sum_type * refcase) {
  
  model_config->forward_model             = forward_model_alloc(  joblist );
  model_config_set_refcase( model_config , refcase );

  if (config_item_set( config , FORWARD_MODEL_KEY )) {
    char * config_string = config_alloc_joined_string( config , FORWARD_MODEL_KEY , " ");
    forward_model_parse_init( model_config->forward_model , config_string );
    free(config_string);
  }
  

  if (config_item_set( config , ENKF_SCHED_FILE_KEY))
    model_config_set_enkf_sched_file(model_config , config_get_value(config , ENKF_SCHED_FILE_KEY ));
  
  if (config_item_set( config, RUNPATH_KEY))
    model_config_set_runpath_fmt( model_config , config_get_value(config , RUNPATH_KEY) );

  if (config_item_set( config, IGNORE_SCHEDULE_KEY))
    model_config->ignore_schedule = config_get_value_as_bool(config , IGNORE_SCHEDULE_KEY);
  
  if (sched_file != NULL) {
    model_config->history              = history_alloc_from_sched_file( SUMMARY_KEY_JOIN_STRING , sched_file , model_config->ignore_schedule);  
    model_config->last_history_restart = last_history_restart;
  }
  
  if (config_item_set( config , HISTORY_SOURCE_KEY)) {
    const char * history_source = config_iget(config , HISTORY_SOURCE_KEY, 0,0);
    bool  use_history;
    history_source_type source_type = history_get_source_type( history_source );
    
    if (source_type == REFCASE_SIMULATED)
      use_history = false;
    else if (source_type == REFCASE_HISTORY)
      use_history = true;
    
    model_config_set_history_source( model_config , source_type );
  }

  if (model_config->history != NULL) {
    int num_restart = history_get_num_restarts( model_config->history );
    bool_vector_iset( model_config->internalize_state , num_restart - 1 , false );
    bool_vector_iset( model_config->__load_state      , num_restart - 1 , false );
  }

  /*
    The full treatment of the SCHEDULE_PREDICTION_FILE keyword is in
    the ensemble_config file, because the functionality is implemented
    as (quite) plain GEN_KW instance. Here we just check if it is
    present or not.
  */
  
  if (config_item_set(config ,  SCHEDULE_PREDICTION_FILE_KEY)) 
    model_config->has_prediction = true;
  else
    model_config->has_prediction = false;


  if (config_item_set(config ,  CASE_TABLE_KEY)) 
    model_config_set_case_table( model_config , ens_size , config_iget( config , CASE_TABLE_KEY , 0,0));
  
  if (config_item_set( config , ENSPATH_KEY))
    model_config_set_enspath( model_config , config_get_value(config , ENSPATH_KEY));
  
  if (config_item_set( config , DBASE_TYPE_KEY))
    model_config_set_dbase_type( model_config , config_get_value(config , DBASE_TYPE_KEY));
  
  if (config_item_set( config , MAX_RESAMPLE_KEY))
    model_config_set_max_resample( model_config , config_get_value_as_int( config , MAX_RESAMPLE_KEY ));

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
  util_safe_free( model_config->select_case );
  util_safe_free( model_config->case_table_file );
  if (model_config->history != NULL)
    history_free(model_config->history);
  forward_model_free(model_config->forward_model);
  bool_vector_free(model_config->internalize_state);
  bool_vector_free(model_config->__load_state);
  if (model_config->case_names != NULL) stringlist_free( model_config->case_names );
  free(model_config);
}


void model_config_set_select_case( model_config_type * model_config , const char * select_case) {
  model_config->select_case = util_realloc_string_copy( model_config->select_case , select_case );
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


forward_model_type * model_config_get_forward_model( const model_config_type * config) {
  return config->forward_model;
}


/*****************************************************************/

/* Setting everything back to the default value: false. */
void model_config_init_internalization( model_config_type * config ) {
  bool_vector_reset(config->internalize_state);
  bool_vector_reset(config->__load_state);
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


void model_config_set_load_state( model_config_type * config , int report_step) {
  bool_vector_iset(config->__load_state , report_step , true);
}



/* Query functions. */

bool model_config_internalize_state( const model_config_type * config , int report_step) {
  return bool_vector_iget(config->internalize_state , report_step);
}

/*****************************************************************/

bool model_config_load_state( const model_config_type * config , int report_step) {
  return bool_vector_iget(config->__load_state , report_step);
}


int model_config_get_max_internal_submit( const model_config_type * config ) {
  return config->max_internal_submit;
}


void model_config_fprintf_config( const model_config_type * model_config , int ens_size , FILE * stream ) {
  fprintf( stream , CONFIG_COMMENTLINE_FORMAT );
  fprintf( stream , CONFIG_COMMENT_FORMAT , "Here comes configuration information related to this model.");

  if (model_config->case_table_file != NULL) {
    fprintf( stream , CONFIG_KEY_FORMAT      , CASE_TABLE_KEY );
    fprintf( stream , CONFIG_ENDVALUE_FORMAT , model_config->case_table_file );
  }
  fprintf( stream , CONFIG_KEY_FORMAT      , FORWARD_MODEL_KEY);  
  forward_model_fprintf( model_config->forward_model , stream );

  if (model_config->ignore_schedule != DEFAULT_IGNORE_SCHEDULE) {
    fprintf( stream , CONFIG_KEY_FORMAT      , IGNORE_SCHEDULE_KEY);  
    fprintf(stream  , CONFIG_ENDVALUE_FORMAT , CONFIG_BOOL_STRING( model_config->ignore_schedule ));
  }

  fprintf( stream , CONFIG_KEY_FORMAT      , RUNPATH_KEY );
  fprintf( stream , CONFIG_ENDVALUE_FORMAT , path_fmt_get_fmt( model_config->runpath ));

  if (model_config->enkf_sched_file != NULL) {
    fprintf( stream , CONFIG_KEY_FORMAT      , ENKF_SCHED_FILE_KEY );
    fprintf( stream , CONFIG_ENDVALUE_FORMAT , model_config->enkf_sched_file );
  }
    
  fprintf( stream , CONFIG_KEY_FORMAT      , ENSPATH_KEY );
  fprintf( stream , CONFIG_ENDVALUE_FORMAT , model_config->enspath );
  
  if (model_config->select_case != NULL) {
    fprintf( stream , CONFIG_KEY_FORMAT      , SELECT_CASE_KEY );
    fprintf( stream , CONFIG_ENDVALUE_FORMAT , model_config->select_case );
  }

  fprintf( stream , CONFIG_KEY_FORMAT      , MAX_RESAMPLE_KEY ); 
  {
    char max_retry_string[16];
    sprintf( max_retry_string , "%d" ,model_config->max_internal_submit);
    fprintf( stream , CONFIG_ENDVALUE_FORMAT , max_retry_string);
  }
  
  fprintf(stream , CONFIG_KEY_FORMAT      , HISTORY_SOURCE_KEY);
  fprintf(stream , CONFIG_ENDVALUE_FORMAT , history_get_source_string( model_config->history_source ));

  fprintf(stream , CONFIG_KEY_FORMAT , NUM_REALIZATIONS_KEY);
  fprintf(stream , CONFIG_INT_FORMAT , ens_size);
  fprintf(stream , "\n\n");

}
