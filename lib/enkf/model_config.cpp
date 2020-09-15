/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'model_config.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>

#include <ert/util/type_macros.h>
#include <ert/util/util.h>
#include <ert/res_util/path_fmt.hpp>
#include <ert/util/hash.h>
#include <ert/util/bool_vector.h>

#include <ert/sched/history.hpp>

#include <ert/config/config_parser.hpp>
#include <ert/config/config_content.hpp>

#include <ert/ecl/ecl_sum.h>
#include <ert/ecl/ecl_util.h>
#include <ert/ecl/ecl_grid.h>

#include <ert/job_queue/forward_model.hpp>

#include <ert/res_util/res_log.hpp>

#include <ert/enkf/model_config.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/fs_types.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/time_map.hpp>
#include <ert/enkf/ert_workflow_list.hpp>
#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/ecl_config.hpp>
#include <ert/enkf/rng_config.hpp>
#include <ert/enkf/hook_manager.hpp>
#include <ert/enkf/site_config.hpp>
#include <ert/enkf/model_config.hpp>

/**
   This struct contains configuration which is specific to this
   particular model/run. Such of the information is actually accessed
   directly through the enkf_state object; but this struct is the
   owner of the information, and responsible for allocating/freeing
   it.

   Observe that the distinction of what goes in model_config, and what
   goes in ecl_config is not entirely clear; ECLIPSE is unfortunately
   not (yet ??) exactly 'any' reservoir simulator in this context.

*/


/*
  The runpath format is governed by a hash table where new runpaths
  are added with model_config_add_runpath() and then current runpath
  is selected with model_config_select_runpath(). However this
  implementation is quite different from the way manipulation of the
  runpath is exposed to the user: The runpath is controlled through
  the RUNPATH config key (key DEFAULT_RUNPATH_KEY in the hash table)
  This semantically predefined runpath is the only option visible to the user.
 */

#define MODEL_CONFIG_TYPE_ID 661053
struct model_config_struct {
  UTIL_TYPE_ID_DECLARATION;
  forward_model_type   * forward_model;             /* The forward_model - as loaded from the config file. Each enkf_state object internalizes its private copy of the forward_model. */
  time_map_type        * external_time_map;
  history_type         * history;                   /* The history object. */
  path_fmt_type        * current_runpath;           /* path_fmt instance for runpath - runtime the call gets arguments: (iens, report_step1 , report_step2) - i.e. at least one %d must be present.*/
  char                 * current_path_key;
  hash_type            * runpath_map;
  char                 * jobname_fmt;
  char                 * enspath;
  char                 * rftpath;
  char                 * data_root;
  char                 * default_data_root;

  fs_driver_impl         dbase_type;
  int                    max_internal_submit;        /* How many times to retry if the load fails. */
  const ecl_sum_type   * refcase;                    /* A pointer to the refcase - can be NULL. Observe that this ONLY a pointer
                                                        to the ecl_sum instance owned and held by the ecl_config object. */
  char                 * gen_kw_export_name;
  int                    num_realizations;
  char                 * obs_config_file;

  /** The results are always loaded. */
  bool_vector_type    * internalize_state;          /* Should the (full) state be internalized (at this report_step). */
  bool_vector_type    * __load_eclipse_restart;     /* Internal variable: is it necessary to load the state? */
};


char * model_config_alloc_jobname( const model_config_type * model_config , int iens) {
  return util_alloc_sprintf( model_config->jobname_fmt, iens);
}

const char * model_config_get_jobname_fmt( const model_config_type * model_config ) {
  return model_config->jobname_fmt;
}

void model_config_set_jobname_fmt( model_config_type * model_config , const char * jobname_fmt) {
  model_config->jobname_fmt = util_realloc_string_copy( model_config->jobname_fmt , jobname_fmt );
}

const char * model_config_get_obs_config_file(const model_config_type * model_config) {
  return model_config->obs_config_file;
}

path_fmt_type * model_config_get_runpath_fmt(const model_config_type * model_config) {
  return model_config->current_runpath;
}

const char * model_config_get_runpath_as_char( const model_config_type * model_config ) {
   return path_fmt_get_fmt( model_config->current_runpath );
}

bool model_config_runpath_requires_iter( const model_config_type * model_config ) {
  if (util_int_format_count( model_config_get_runpath_as_char( model_config)) > 1 )
    return true;
  else
    return false;
}




void model_config_add_runpath( model_config_type * model_config , const char * path_key , const char * fmt) {
  path_fmt_type * path_fmt = path_fmt_alloc_directory_fmt( fmt );
  hash_insert_hash_owned_ref( model_config->runpath_map , path_key , path_fmt , path_fmt_free__ );
}


/*
  If the path_key does not exists it will return false and stay
  silent.
*/

bool model_config_select_runpath( model_config_type * model_config , const char * path_key) {
  if (hash_has_key( model_config->runpath_map , path_key )) {
    model_config->current_runpath = (path_fmt_type * ) hash_get( model_config->runpath_map , path_key );
    if(model_config->current_path_key != path_key) // If ptrs are the same, there is nothing to do
        model_config->current_path_key = util_realloc_string_copy( model_config->current_path_key , path_key);
    return true;
  } else {
    if (model_config->current_runpath != NULL)  // OK - we already have a valid selection - stick to that and return False.
      return false;
    else {
      util_abort("%s: path_key:%s does not exist - and currently no valid runpath selected \n",__func__ , path_key);
      return false;
    }
  }
}


void model_config_set_runpath(model_config_type * model_config , const char * fmt) {
  if (model_config->current_path_key) {
    model_config_add_runpath(model_config , model_config->current_path_key , fmt);
    model_config_select_runpath( model_config , model_config->current_path_key );
  } else
    util_abort("%s: current path has not been set \n",__func__);
}



void  model_config_set_gen_kw_export_name( model_config_type * model_config, const char * name) {
  model_config->gen_kw_export_name = util_realloc_string_copy( model_config->gen_kw_export_name , name );
}

const char * model_config_get_gen_kw_export_name( const model_config_type * model_config) {
  return model_config->gen_kw_export_name;
}



 void model_config_set_enspath( model_config_type * model_config , const char * enspath) {
   model_config->enspath = util_realloc_string_copy( model_config->enspath , enspath );
 }

 void model_config_set_rftpath( model_config_type * model_config , const char * rftpath) {
   model_config->rftpath = util_realloc_string_copy( model_config->rftpath , rftpath );
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

const ecl_sum_type * model_config_get_refcase( const model_config_type * model_config ) {
  return model_config->refcase;
}

void * model_config_get_dbase_args( const model_config_type * model_config ) {
  return NULL;
}


void model_config_set_refcase( model_config_type * model_config , const ecl_sum_type * refcase ) {
  model_config->refcase = refcase;
}


history_source_type model_config_get_history_source( const model_config_type * model_config ) {
  if(!model_config->history)
    return HISTORY_SOURCE_INVALID;

  return history_get_source(model_config->history);
}


void model_config_select_refcase_history( model_config_type * model_config , const ecl_sum_type * refcase , bool use_history) {
  if (model_config->history != NULL)
    history_free( model_config->history );

  if (refcase != NULL) {
    model_config->history = history_alloc_from_refcase( refcase , use_history );
  } else
    util_abort("%s: internal error - trying to load history from REFCASE - but no REFCASE has been loaded.\n",__func__);
}


int model_config_get_max_internal_submit( const model_config_type * config ) {
  return config->max_internal_submit;
}

void model_config_set_max_internal_submit( model_config_type * model_config , int max_resample ) {
  model_config->max_internal_submit = max_resample;
}


UTIL_IS_INSTANCE_FUNCTION( model_config , MODEL_CONFIG_TYPE_ID)

model_config_type * model_config_alloc_empty() {
  model_config_type * model_config = (model_config_type *)util_malloc(sizeof * model_config );
  /**
     There are essentially three levels of initialisation:

     1. Initialize to NULL / invalid.
     2. Initialize with default values.
     3. Initialize with user supplied values.

  */
  UTIL_TYPE_ID_INIT(model_config , MODEL_CONFIG_TYPE_ID);
  model_config->enspath                   = NULL;
  model_config->rftpath                   = NULL;
  model_config->data_root                 = NULL;
  model_config->default_data_root         = NULL;
  model_config->dbase_type                = INVALID_DRIVER_ID;
  model_config->current_runpath           = NULL;
  model_config->current_path_key          = NULL;
  model_config->history                   = NULL;
  model_config->jobname_fmt               = NULL;
  model_config->forward_model             = NULL;
  model_config->external_time_map         = NULL;
  model_config->internalize_state         = bool_vector_alloc( 0 , false );
  model_config->__load_eclipse_restart    = bool_vector_alloc( 0 , false );
  model_config->runpath_map               = hash_alloc();
  model_config->gen_kw_export_name        = NULL;
  model_config->refcase                   = NULL;
  model_config->num_realizations          = 0;
  model_config->obs_config_file           = NULL;

  model_config_set_enspath( model_config        , DEFAULT_ENSPATH );
  model_config_set_rftpath( model_config        , DEFAULT_RFTPATH );
  model_config_set_dbase_type( model_config     , DEFAULT_DBASE_TYPE );
  model_config_set_max_internal_submit( model_config   , DEFAULT_MAX_INTERNAL_SUBMIT);
  model_config_add_runpath( model_config , DEFAULT_RUNPATH_KEY , DEFAULT_RUNPATH);
  model_config_select_runpath( model_config , DEFAULT_RUNPATH_KEY );
  model_config_set_gen_kw_export_name(model_config, DEFAULT_GEN_KW_EXPORT_NAME);

  return model_config;
}

model_config_type * model_config_alloc(
        const config_content_type * config_content,
        const char * data_root,
        const ext_joblist_type * joblist,
        int last_history_restart,
        const ecl_sum_type * refcase)
{
  model_config_type * model_config = model_config_alloc_empty();

  if(config_content)
    model_config_init(model_config,
                      config_content,
                      data_root,
                      0,
                      joblist,
                      last_history_restart,
                      refcase);

  return model_config;
}

model_config_type * model_config_alloc_full(int max_resample,
                                           int num_realizations,
                                           char * run_path,
                                           char * data_root,
                                           char * enspath,
                                           char * job_name,
                                           forward_model_type * forward_model,
                                           char * obs_config,
                                           time_map_type * time_map,
                                           char * rftpath,
                                           char * gen_kw_export_name,
                                           history_source_type history_source,
                                           const ext_joblist_type * joblist,
                                           const ecl_sum_type * refcase)
{
  model_config_type * model_config = model_config_alloc_empty();
  model_config->max_internal_submit = max_resample;
  model_config->num_realizations = num_realizations;
  model_config->dbase_type = BLOCK_FS_DRIVER_ID;
  
  model_config_add_runpath( model_config , DEFAULT_RUNPATH_KEY , run_path);
  model_config_select_runpath( model_config , DEFAULT_RUNPATH_KEY );
  model_config_set_data_root(model_config, data_root);

  model_config->enspath = util_realloc_string_copy( model_config->enspath , enspath );
  model_config->jobname_fmt = util_realloc_string_copy( model_config->jobname_fmt , job_name );
  model_config->forward_model = forward_model;
  model_config->obs_config_file = util_alloc_string_copy(obs_config);
  model_config->external_time_map = time_map;
  model_config->rftpath = util_realloc_string_copy( model_config->rftpath , rftpath );
  model_config->gen_kw_export_name = util_realloc_string_copy( model_config->gen_kw_export_name , gen_kw_export_name );
  model_config->refcase = refcase;
  
  model_config_select_history(model_config, history_source, refcase);

  if (model_config->history != NULL) {
    int num_restart = model_config_get_last_history_restart(model_config);
    bool_vector_iset( model_config->internalize_state , num_restart - 1 , false );
    bool_vector_iset( model_config->__load_eclipse_restart      , num_restart - 1 , false );
  }
  
  return model_config;
}


bool model_config_select_history( model_config_type * model_config , history_source_type source_type,  const ecl_sum_type * refcase) {
  bool selectOK = false;

  if (((source_type == REFCASE_HISTORY) || (source_type == REFCASE_SIMULATED)) && refcase != NULL) {
    if (source_type == REFCASE_HISTORY)
      model_config_select_refcase_history( model_config , refcase , true);
    else
      model_config_select_refcase_history( model_config , refcase , false);
    selectOK = true;
  }

  return selectOK;
}


static bool model_config_select_any_history( model_config_type * model_config , const ecl_sum_type * refcase) {
  bool selectOK = false;

  if ( refcase != NULL ) {
    model_config_select_refcase_history( model_config , refcase , true);
    selectOK = true;
  }

  return selectOK;
}

const char * model_config_get_data_root( const model_config_type * model_config ) {
  if (model_config->data_root)
    return model_config->data_root;

  return model_config->default_data_root;
}

void model_config_set_data_root( model_config_type * model_config , const char * data_root) {
  model_config->data_root = util_realloc_string_copy( model_config->data_root , data_root );
  setenv( "DATA_ROOT" ,  data_root , 1 );
}

static void model_config_set_default_data_root( model_config_type * model_config , const char * data_root) {
  model_config->default_data_root = util_alloc_string_copy( data_root );
  setenv( "DATA_ROOT" ,  data_root , 1 );
}


bool model_config_data_root_is_set( const model_config_type * model_config ) {
  if (!model_config->data_root)
    return false;

  return !util_string_equal( model_config->data_root, model_config->default_data_root);
}



void model_config_init(model_config_type * model_config ,
                       const config_content_type * config ,
                       const char * data_root,
                       int ens_size ,
                       const ext_joblist_type * joblist ,
                       int last_history_restart ,
                       const ecl_sum_type * refcase) {

  model_config->forward_model = forward_model_alloc(  joblist );
  model_config_set_refcase( model_config , refcase );
  model_config_set_default_data_root( model_config, data_root );

  if (config_content_has_item(config, NUM_REALIZATIONS_KEY))
    model_config->num_realizations = config_content_get_value_as_int(config, NUM_REALIZATIONS_KEY);

  for (int i = 0; i < config_content_get_size(config); i++) {
    const config_content_node_type * node = config_content_iget_node( config , i);
    if (util_string_equal(config_content_node_get_kw(node), SIMULATION_JOB_KEY))
      forward_model_parse_job_args(model_config->forward_model,
                                   config_content_node_get_stringlist(node));

    if (util_string_equal(config_content_node_get_kw(node), FORWARD_MODEL_KEY) ) {
      const char * arg = config_content_node_get_full_string(node, "");
      forward_model_parse_job_deprecated_args( model_config->forward_model , arg );
    }
  }


  if (config_content_has_item( config, RUNPATH_KEY)) {
    model_config_add_runpath( model_config , DEFAULT_RUNPATH_KEY , config_content_get_value_as_path(config , RUNPATH_KEY) );
    model_config_select_runpath( model_config , DEFAULT_RUNPATH_KEY );
  }

  history_source_type source_type = DEFAULT_HISTORY_SOURCE;

  if (config_content_has_item( config , HISTORY_SOURCE_KEY)) {
    const char * history_source = config_content_iget(config , HISTORY_SOURCE_KEY, 0,0);
    source_type = history_get_source_type( history_source );
  }

  if (!model_config_select_history( model_config , source_type  , refcase ))
    if (!model_config_select_history( model_config , DEFAULT_HISTORY_SOURCE , refcase )) {
      model_config_select_any_history( model_config , refcase);
      /* If even the last call return false, it means the configuration does not have any of
       * these keys: HISTORY_SOURCE or REFCASE.
       * History matching won't be supported for this configuration.
       */
    }

  if (model_config->history != NULL) {
    int num_restart = model_config_get_last_history_restart(model_config);
    bool_vector_iset( model_config->internalize_state , num_restart - 1 , false );
    bool_vector_iset( model_config->__load_eclipse_restart      , num_restart - 1 , false );
  }

  if (config_content_has_item( config , TIME_MAP_KEY)) {
    const char * filename = config_content_get_value_as_path( config , TIME_MAP_KEY);
    time_map_type * time_map = time_map_alloc();
    if (time_map_fscanf( time_map , filename))
      model_config->external_time_map = time_map;
    else {
      time_map_free( time_map );
      fprintf(stderr,"** ERROR: Loading external time map from:%s failed \n", filename);
    }
  }



  /*
    The full treatment of the SCHEDULE_PREDICTION_FILE keyword is in
    the ensemble_config file, because the functionality is implemented
    as (quite) plain GEN_KW instance. Here we just check if it is
    present or not.
  */

  if (config_content_has_item( config , ENSPATH_KEY))
    model_config_set_enspath( model_config , config_content_get_value_as_path(config , ENSPATH_KEY));

  if (config_content_has_item( config , DATA_ROOT_KEY))
    model_config_set_data_root( model_config , config_content_get_value_as_path(config , DATA_ROOT_KEY));

  /*
    The keywords ECLBASE and JOBNAME can be used as synonyms. But
    observe that:

      1. The ecl_config object will also pick up the ECLBASE keyword,
         and set the have_eclbase flag of that object.

      2. If both ECLBASE and JOBNAME are in the config file the
         JOBNAME keyword will be preferred.
  */
  if (config_content_has_item( config , ECLBASE_KEY))
    model_config_set_jobname_fmt( model_config , config_content_get_value(config , ECLBASE_KEY));

  if (config_content_has_item( config , JOBNAME_KEY)) {
    model_config_set_jobname_fmt( model_config , config_content_get_value(config , JOBNAME_KEY));
    if (config_content_has_item( config , ECLBASE_KEY))
      res_log_warning("Can not have both JOBNAME and ECLBASE keywords. The ECLBASE keyword will be ignored.");
  }

  if (config_content_has_item( config , RFTPATH_KEY))
    model_config_set_rftpath( model_config , config_content_get_value(config , RFTPATH_KEY));

  if (config_content_has_item( config , MAX_RESAMPLE_KEY))
    model_config_set_max_internal_submit( model_config , config_content_get_value_as_int( config , MAX_RESAMPLE_KEY ));


  {
    if (config_content_has_item( config , GEN_KW_EXPORT_NAME_KEY)) {
      const char * export_name = config_content_get_value(config, GEN_KW_EXPORT_NAME_KEY);
      model_config_set_gen_kw_export_name(model_config, export_name);
    }
  }

  if (config_content_has_item(config, OBS_CONFIG_KEY)) {
    const char * obs_config_file = config_content_get_value_as_abspath(
                                                    config,
                                                    OBS_CONFIG_KEY
                                                    );

    model_config->obs_config_file = util_alloc_string_copy(obs_config_file);
  }
}





void model_config_free(model_config_type * model_config) {
  free( model_config->enspath );
  free( model_config->rftpath );
  free( model_config->jobname_fmt );
  free( model_config->current_path_key);
  free( model_config->gen_kw_export_name);
  free( model_config->obs_config_file );
  free( model_config->data_root );
  free( model_config->default_data_root );

  if (model_config->history)
    history_free(model_config->history);

  if (model_config->forward_model)
    forward_model_free(model_config->forward_model);

  if (model_config->external_time_map)
    time_map_free( model_config->external_time_map );

  bool_vector_free(model_config->internalize_state);
  bool_vector_free(model_config->__load_eclipse_restart);
  hash_free(model_config->runpath_map);
  free(model_config);
}



bool model_config_has_history(const model_config_type * config) {
  if (config->history != NULL)
    return true;
  else
    return false;
}


history_type * model_config_get_history(const model_config_type * config) {
  return config->history;
}

int model_config_get_num_realizations(const model_config_type * model_config) {
  return model_config->num_realizations;
}

/**
   Will be NULL unless the user has explicitly loaded an external time
   map with the TIME_MAP config option.
*/

time_map_type * model_config_get_external_time_map( const model_config_type * config) {
  return config->external_time_map;
}

int model_config_get_last_history_restart(const model_config_type * config) {
  if (config->history)
    return history_get_last_restart( config->history );
  else {
    if (config->external_time_map)
      return time_map_get_last_step( config->external_time_map);
    else
      return -1;
  }
}

forward_model_type * model_config_get_forward_model( const model_config_type * config) {
  return config->forward_model;
}


/*****************************************************************/

/* Setting everything back to the default value: false. */
void model_config_init_internalization( model_config_type * config ) {
  bool_vector_reset(config->internalize_state);
  bool_vector_reset(config->__load_eclipse_restart);
}


/**
   This function sets the internalize_state flag to true for
   report_step. Because of the coupling to the __load_eclipse_restart variable
   this function can __ONLY__ be used to set internalize to true.
*/

void model_config_set_internalize_state( model_config_type * config , int report_step) {
  bool_vector_iset(config->internalize_state , report_step , true);
  bool_vector_iset(config->__load_eclipse_restart      , report_step , true);
}


/*****************************************************************/

static char * model_config_alloc_user_config_file(const char * user_config_file, bool base_only) {
    char * base_name;
    char * extension;
    util_alloc_file_components(user_config_file, NULL, &base_name, &extension);

    char * config_file;
    if (base_only)
      config_file = util_alloc_filename(NULL, base_name, NULL);
    else
      config_file = util_alloc_filename(NULL, base_name, extension);

    free(base_name);
    free(extension);

    return config_file;
}

static hash_type * alloc_predefined_kw_map(const char * user_config_file) {
    char * config_file_base       = model_config_alloc_user_config_file(user_config_file, true);
    char * config_file            = model_config_alloc_user_config_file(user_config_file, false);
    char * config_path;

    hash_type * pre_defined_kw_map = hash_alloc();
    hash_insert_string(pre_defined_kw_map, "<CONFIG_FILE>", config_file);
    hash_insert_string(pre_defined_kw_map, "<CONFIG_FILE_BASE>", config_file_base);
    {
      char * tmp_path;
      util_alloc_file_components( user_config_file , &tmp_path , NULL , NULL );
      config_path = util_alloc_abs_path( tmp_path );
      free( tmp_path );
    }
    hash_insert_string(pre_defined_kw_map, "<CONFIG_PATH>", config_path);
    free(config_path);
    free(config_file) ;
    free(config_file_base);

    return pre_defined_kw_map;
}

static void model_config_init_user_config(config_parser_type * config ) {
  config_schema_item_type * item;

  /*****************************************************************/
  /* config_add_schema_item():                                     */
  /*                                                               */
  /*  1. boolean - required?                                       */
  /*****************************************************************/

  ert_workflow_list_add_config_items( config );
  analysis_config_add_config_items( config );
  ensemble_config_add_config_items( config );
  ecl_config_add_config_items( config );
  rng_config_add_config_items( config );

  /*****************************************************************/
  /* Required keywords from the ordinary model_config file */

  config_add_key_value(config, LOG_LEVEL_KEY, false, CONFIG_STRING);
  config_add_key_value(config, LOG_FILE_KEY, false, CONFIG_PATH);

  config_add_key_value(config, MAX_RESAMPLE_KEY, false, CONFIG_INT);


  item = config_add_schema_item(config, NUM_REALIZATIONS_KEY, true);
  config_schema_item_set_argc_minmax(item, 1, 1);
  config_schema_item_iset_type(item, 0, CONFIG_INT);
  config_add_alias(config, NUM_REALIZATIONS_KEY, "SIZE");
  config_add_alias(config, NUM_REALIZATIONS_KEY, "NUM_REALISATIONS");
  config_install_message(
          config, "SIZE",
          "** Warning: \'SIZE\' is depreceated "
          "- use \'NUM_REALIZATIONS\' instead."
          );


  /*****************************************************************/
  /* Optional keywords from the model config file */

  item = config_add_schema_item(config, RUN_TEMPLATE_KEY, false);
  config_schema_item_set_argc_minmax(item, 2, CONFIG_DEFAULT_ARG_MAX);
  config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);

  config_add_key_value(config, RUNPATH_KEY, false, CONFIG_PATH);
  config_add_key_value(config, DATA_ROOT_KEY, false, CONFIG_PATH);
  config_add_key_value(config, ENSPATH_KEY, false, CONFIG_PATH);

  item = config_add_schema_item(config, JOBNAME_KEY, false);
  config_schema_item_set_argc_minmax(item, 1, 1);

  item = config_add_schema_item(config, DBASE_TYPE_KEY, false);
  config_parser_deprecate(
    config, DBASE_TYPE_KEY,
    "\'DBASE_TYPE\' has been deprecated."
  );

  item = config_add_schema_item(config, FORWARD_MODEL_KEY, false);
  config_schema_item_set_argc_minmax(item , 1, CONFIG_DEFAULT_ARG_MAX);

  item = config_add_schema_item(config, SIMULATION_JOB_KEY, false);
  config_schema_item_set_argc_minmax(item , 1, CONFIG_DEFAULT_ARG_MAX);

  item = config_add_schema_item(config, DATA_KW_KEY, false);
  config_schema_item_set_argc_minmax(item, 2, 2);

  item = config_add_schema_item(config, OBS_CONFIG_KEY, false);
  config_schema_item_set_argc_minmax(item , 1, 1);
  config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);

  config_add_key_value(config, TIME_MAP_KEY, false, CONFIG_EXISTING_PATH);

  item = config_add_schema_item(config, RFTPATH_KEY, false);
  config_schema_item_set_argc_minmax(item, 1, 1);

  item = config_add_schema_item(config, GEN_KW_EXPORT_NAME_KEY, false);
  config_schema_item_set_argc_minmax(item, 1, 1);

  item = config_add_schema_item(config, GEN_KW_EXPORT_FILE_KEY, false);
  config_schema_item_set_argc_minmax(item, 1, 1);
  {
    char * msg = util_alloc_sprintf("The keyword:%s has been deprecated - use %s *WITHOUT* extension instead", GEN_KW_EXPORT_FILE_KEY, GEN_KW_EXPORT_NAME_KEY);
    config_parser_deprecate( config, GEN_KW_EXPORT_FILE_KEY, msg );
    free( msg );
  }


  item = config_add_schema_item(config, LOCAL_CONFIG_KEY, false);
  config_parser_deprecate(
    config, LOCAL_CONFIG_KEY,
    "\'LOCAL_CONFIG\' is deprecated."
  );

  stringlist_type * refcase_dep = stringlist_alloc_new();
  stringlist_append_copy(refcase_dep, REFCASE_KEY);
  item = config_add_schema_item(config, HISTORY_SOURCE_KEY, false);
  config_schema_item_set_argc_minmax(item , 1 , 1);
  {
    stringlist_type * argv = stringlist_alloc_new();
    stringlist_append_copy(argv, "SCHEDULE");
    stringlist_append_copy(argv, "REFCASE_SIMULATED");
    stringlist_append_copy(argv, "REFCASE_HISTORY");

    config_schema_item_set_common_selection_set(item, argv );
    stringlist_free(argv);
  }
  config_schema_item_set_required_children_on_value(item, "REFCASE_SIMULATED", refcase_dep);
  config_schema_item_set_required_children_on_value(item, "REFCASE_HISTORY", refcase_dep);

  stringlist_free(refcase_dep);

  hook_manager_add_config_items(config);
}


void model_config_init_config_parser(config_parser_type * config_parser) {
  model_config_init_user_config(config_parser);
  site_config_add_config_items(config_parser, false);
}

config_content_type * model_config_alloc_content(
        const char * user_config_file, config_parser_type * config) {

  model_config_init_config_parser(config);

  hash_type * pre_defined_kw_map = alloc_predefined_kw_map(user_config_file);
  config_content_type * content = config_parse(
          config , user_config_file,
          "--", INCLUDE_KEY, DEFINE_KEY,
          pre_defined_kw_map, CONFIG_UNRECOGNIZED_WARN, true
          );
  hash_free(pre_defined_kw_map);

  const stringlist_type * warnings = config_content_get_warnings(content);
  if (stringlist_get_size( warnings ) > 0) {
    fprintf(
            stderr,
            " ** There were warnings when parsing the configuration file: %s",
            user_config_file
            );

    for (int i=0; i < stringlist_get_size( warnings ); i++)
      fprintf(stderr, " %02d : %s \n", i, stringlist_iget(warnings, i));
  }

  if (!config_content_is_valid(content)) {
    config_error_type * errors = config_content_get_errors(content);
    config_error_fprintf(errors, true, stderr);

    util_abort(
            "%s: Failed to load user configuration file: %s\n",
            __func__, user_config_file
            );
  }

  return content;
}



bool model_config_report_step_compatible(const model_config_type * model_config, const ecl_sum_type * ecl_sum_simulated) {
  bool ret = true;
  const ecl_sum_type * ecl_sum_reference = model_config_get_refcase(model_config);

  if (ecl_sum_reference) //Can be NULL
    ret = ecl_sum_report_step_compatible(ecl_sum_reference, ecl_sum_simulated);

  return ret;
}
