#include <enkf_util.h>
#include <time.h>
#include <util.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <ecl_io_config.h>
#include <set.h>
#include <path_fmt.h>
#include <ecl_grid.h>
#include <sched_file.h>
#include <config.h>
#include <ecl_config.h>


#include "enkf_defaults.h"


/**
  This file implements a struct which holds configuration information
  needed to run ECLIPSE.

  Pointers to the fields in this structure are passed on to e.g. the
  enkf_state->shared_info object, but this struct is the *OWNER* of
  this information, and hence responsible for booting and deleting
  these objects.

   Observe that the distinction of what goes in model_config, and what
   goes in ecl_config is not entirely clear.
*/


struct ecl_config_struct {
  ecl_io_config_type * io_config;              	   /* This struct contains information of whether the eclipse files should be formatted|unified|endian_fliped */
  path_fmt_type      * eclbase;                	   /* A pth_fmt instance with one %d specifer which will be used for eclbase - members will allocate private eclbase; i.e. updates will not be refelected. */
  sched_file_type    * sched_file;             	   /* Will only contain the history - if predictions are active the member_config objects will have a private sched_file instance. */
  path_fmt_type      * prediction_sched_file_fmt;  /* A format variable for schedule prediction files - can be NULL. */
  bool                 include_all_static_kw;  	   /* If true all static keywords are stored.*/ 
  set_type           * static_kw_set;          	   /* Minimum set of static keywords which must be included to make valid restart files. */
  char               * data_file;              	   /* Eclipse data file. */
  ecl_grid_type      * grid;                   	   /* The grid which is active for this model. */
  char               * schedule_target_file;   	   /* File name to write schedule info to */
  char               * equil_init_file;        	   /* File name for ECLIPSE (EQUIL) initialisation. */
};


/*****************************************************************/



ecl_config_type * ecl_config_alloc( const config_type * config , int * history_length) {
  ecl_config_type * ecl_config      = util_malloc(sizeof * ecl_config , __func__);
  ecl_config->io_config 	    = ecl_io_config_alloc( DEFAULT_FORMATTED , DEFAULT_ENDIAN_FLIP , DEFAULT_UNIFIED );
  ecl_config->eclbase   	    = path_fmt_alloc_path_fmt( config_get(config , "ECLBASE") );
  ecl_config->include_all_static_kw = false;
  ecl_config->static_kw_set         = set_alloc_empty();
  {
    for (int ikw = 0; ikw < NUM_STATIC_KW; ikw++)
      set_add_key(ecl_config->static_kw_set , DEFAULT_STATIC_KW[ikw]);
  }
  ecl_config->data_file = util_alloc_string_copy(config_get( config , "DATA_FILE" ));
  {
    time_t start_date = ecl_util_get_start_date( ecl_config->data_file );
    const stringlist_type * sched_list = config_get_stringlist_ref(config , "SCHEDULE_FILE");
    const char * schedule_src = stringlist_iget( sched_list , 0);

    {
      char * base;  /* The schedule target file will be without any path component */
      char * ext;
      util_alloc_file_components(schedule_src , NULL , &base , &ext);
      ecl_config->schedule_target_file = util_alloc_filename(NULL , base , ext);
      free(ext);
      free(base);
    } 

    ecl_config->sched_file = sched_file_parse_alloc( schedule_src , start_date );
    *history_length = sched_file_get_num_restart_files( ecl_config->sched_file );   /* We keep track of this - so we can stop assimilation at the
												  end of HISTORY. */
    if (config_has_set_item(config , "SCHEDULE_PREDICTION_FILE"))
      ecl_config->prediction_sched_file_fmt = path_fmt_alloc_path_fmt( config_get(config , "SCHEDULE_PREDICTION_FILE") );
    else
      ecl_config->prediction_sched_file_fmt = NULL;

  }
  if (config_has_set_item(config , "EQUIL_INIT_FILE"))
    ecl_config->equil_init_file = util_alloc_realpath(config_get(config , "EQUIL_INIT_FILE"));
  else {
    if (!config_has_set_item(config , "EQUIL"))
      util_abort("%s: you must specify how ECLIPSE is initialized - with either EQUIL or EQUIL_INIT_FILE ",__func__);
    ecl_config->equil_init_file = NULL; 
  }
  ecl_config->grid = ecl_grid_alloc( config_get(config , "GRID") , ecl_io_config_get_endian_flip(ecl_config->io_config) );
  return ecl_config;
}


void ecl_config_free(ecl_config_type * ecl_config) {
  ecl_io_config_free( ecl_config->io_config );
  ecl_grid_free( ecl_config->grid );
  path_fmt_free( ecl_config->eclbase );
  set_free( ecl_config->static_kw_set );
  free(ecl_config->data_file);
  sched_file_free(ecl_config->sched_file);
  free(ecl_config->schedule_target_file);
  util_safe_free(ecl_config->equil_init_file);
  if (ecl_config->prediction_sched_file_fmt != NULL)
    path_fmt_free(ecl_config->prediction_sched_file_fmt);
  
  free(ecl_config);
}



/**
   This function adds a keyword to the list of restart keywords wich
   are included. Observe that ecl_util_escape_kw() is called prior to
   adding it.
   
   The kw __ALL__ is magic; and will result in a request to store all
   static kewyords. This wastes disk-space, but might be beneficial
   when debugging.
*/

void ecl_config_add_static_kw(ecl_config_type * ecl_config , const char * _kw) {
  if (strcmp(_kw , DEFAULT_ALL_STATIC_KW) == 0) 
    ecl_config->include_all_static_kw = true;
  else {
    char * kw = util_alloc_string_copy(_kw);
    ecl_util_escape_kw(kw);
    set_add_key(ecl_config->static_kw_set , kw);
    free(kw);
  }
}




/**
   This function checks whether the static kw should be
   included. Observe that it is __assumed__ that ecl_util_escape_kw()
   has already been called on the kw.
*/


bool ecl_config_include_static_kw(const ecl_config_type * ecl_config, const char * kw) {
  if (ecl_config->include_all_static_kw)
    return true;
  else
    return set_has_key(ecl_config->static_kw_set , kw);
}


const ecl_grid_type * ecl_config_get_grid(const ecl_config_type * ecl_config) {
  return ecl_config->grid;
}


ecl_io_config_type * ecl_config_get_io_config(const ecl_config_type * ecl_config) {
  return ecl_config->io_config;
}


const path_fmt_type * ecl_config_get_eclbase_fmt(const ecl_config_type * ecl_config) {
  return ecl_config->eclbase;
}

sched_file_type * ecl_config_get_sched_file(const ecl_config_type * ecl_config) {
  return ecl_config->sched_file;
}

char * ecl_config_alloc_schedule_prediction_file(const ecl_config_type * ecl_config, int iens) {
  if (ecl_config->prediction_sched_file_fmt != NULL)
    return path_fmt_alloc_path(ecl_config->prediction_sched_file_fmt , false , iens);
  else
    return NULL;
}


const char * ecl_config_get_data_file(const ecl_config_type * ecl_config) {
  return ecl_config->data_file;
}

const char * ecl_config_get_equil_init_file(const ecl_config_type * ecl_config) {
  return ecl_config->equil_init_file;
}

const char * ecl_config_get_schedule_target(const ecl_config_type * ecl_config) {
  return ecl_config->schedule_target_file;
}

int ecl_config_get_num_restart_files(const ecl_config_type * ecl_config) {
  return sched_file_get_num_restart_files(ecl_config->sched_file);
}

bool ecl_config_get_endian_flip(const ecl_config_type * ecl_config) { return ecl_io_config_get_endian_flip(ecl_config->io_config); }
bool ecl_config_get_formatted(const ecl_config_type * ecl_config) { return ecl_io_config_get_formatted(ecl_config->io_config); }
bool ecl_config_get_unified(const ecl_config_type * ecl_config) { return ecl_io_config_get_unified(ecl_config->io_config); }

