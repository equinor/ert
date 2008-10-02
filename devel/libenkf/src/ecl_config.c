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
  ecl_io_config_type * io_config;              /* This struct contains information of whether the eclipse files should be formatted|unified|endian_fliped */
  
  path_fmt_type      * eclbase;                /* A pth_fmt instance with one %d specifer which will be used for eclbase - members will allocate private eclbase; i.e. updates will not be refelected. */
  ecl_grid_type      * grid;                   /* Eclipse grid instance */
  sched_file_type    * sched_file;
  bool                 include_all_static_kw;  /* If true all static keywords are stored.*/ 
  set_type           * static_kw_set;          /* Minimum set of static keywords which must be included to make valid restart files. */
  char               * data_file;              /* Eclipse data file. */
};


/*****************************************************************/



ecl_config_type * ecl_config_alloc( const config_type * config , time_t * start_date) {
  ecl_config_type * ecl_config      = util_malloc(sizeof * ecl_config , __func__);
  ecl_config->io_config 	    = ecl_io_config_alloc( DEFAULT_FORMATTED , DEFAULT_ENDIAN_FLIP , DEFAULT_UNIFIED );
  ecl_config->grid      	    = ecl_grid_alloc( config_get(config , "GRID") , ecl_io_config_get_endian_flip(ecl_config->io_config) );
  ecl_config->eclbase   	    = path_fmt_alloc_path_fmt( config_get(config , "ECLBASE") );
  ecl_config->include_all_static_kw = false;
  ecl_config->static_kw_set         = set_alloc_empty();
  {
    for (int ikw = 0; ikw < NUM_STATIC_KW; ikw++)
      set_add_key(ecl_config->static_kw_set , DEFAULT_STATIC_KW[ikw]);
  }
  ecl_config->data_file = util_alloc_string_copy(config_get( config , "DATA_FILE" ));
  *start_date = ecl_util_get_start_date( ecl_config->data_file );

  ecl_config->sched_file = sched_file_parse_alloc( *start_date , config_iget( config , "SCHEDULE_FILE" , 0) );
  return ecl_config;
}


void ecl_config_free(ecl_config_type * ecl_config) {
  ecl_io_config_free( ecl_config->io_config );
  ecl_grid_free( ecl_config->grid );
  path_fmt_free( ecl_config->eclbase );
  set_free( ecl_config->static_kw_set );
  free(ecl_config->data_file);
  sched_file_free(ecl_config->sched_file);
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

