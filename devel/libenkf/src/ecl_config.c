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
#include <parser.h>

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
  bool                 include_all_static_kw;  	   /* If true all static keywords are stored.*/ 
  set_type           * static_kw_set;          	   /* Minimum set of static keywords which must be included to make valid restart files. */
  char               * data_file;              	   /* Eclipse data file. */
  ecl_grid_type      * grid;                   	   /* The grid which is active for this model. */
  char               * schedule_target_file;   	   /* File name to write schedule info to */
  char               * equil_init_file;        	   /* File name for ECLIPSE (EQUIL) initialisation - can be NULL if the user has not supplied INIT_SECTION. */
  int                  last_history_restart;
  bool                 can_restart;                /* Have we found the <INIT> tag in the data file? */
};


/*****************************************************************/


/**
   Could look up the sched_file instance directly - because the
   ecl_config will never be the owner of a file with predictions.
*/

int ecl_config_get_last_history_restart( const ecl_config_type * ecl_config ) {
  return ecl_config->last_history_restart;
}


bool ecl_config_can_restart( const ecl_config_type * ecl_config ) {
  return ecl_config->can_restart;
}


void ecl_config_set_data_file( ecl_config_type * ecl_config , const char * data_file) {
  ecl_config->data_file = util_realloc_string_copy( ecl_config->data_file , data_file );
  {
    FILE * stream        = util_fopen( ecl_config->data_file , "r");
    parser_type * parser = parser_alloc(NULL , NULL , NULL , NULL , "--" , "\n" );
    char * init_tag      = enkf_util_alloc_tagged_string( "INIT" );
    
    ecl_config->can_restart = parser_fseek_string( parser , stream , init_tag , false , true ); 
    
    free( init_tag );
    parser_free( parser );
    fclose( stream );
  }
}


ecl_config_type * ecl_config_alloc( const config_type * config ) {
  ecl_config_type * ecl_config      = util_malloc(sizeof * ecl_config , __func__);
  ecl_config->io_config 	    = ecl_io_config_alloc( DEFAULT_FORMATTED , DEFAULT_UNIFIED , DEFAULT_UNIFIED );
  ecl_config->eclbase   	    = path_fmt_alloc_path_fmt( config_iget(config , "ECLBASE" ,0,0) );
  ecl_config->include_all_static_kw = false;
  ecl_config->static_kw_set         = set_alloc_empty();
  ecl_config->data_file             = NULL;
  ecl_config->can_restart           = false;
  {
    for (int ikw = 0; ikw < NUM_STATIC_KW; ikw++)
      set_add_key(ecl_config->static_kw_set , DEFAULT_STATIC_KW[ikw]);
  }
  ecl_config_set_data_file( ecl_config , config_iget( config , "DATA_FILE" ,0,0));
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

    ecl_config->sched_file = sched_file_alloc( start_date );
    {
      if (config_has_set_item( config , "ADD_FIXED_LENGTH_SCHEDULE_KW")) {
        int iocc;
        for (iocc = 0; iocc < config_get_occurences(config , "ADD_FIXED_LENGTH_SCHEDULE_KW"); iocc++) 
          sched_file_add_fixed_length_kw( ecl_config->sched_file , 
                                          config_iget(config , "ADD_FIXED_LENGTH_SCHEDULE_KW" , iocc , 0) , 
                                          config_iget_as_int(config , "ADD_FIXED_LENGTH_SCHEDULE_KW" , iocc , 1));
        
      }
    }
    sched_file_parse(ecl_config->sched_file , schedule_src );
    ecl_config->last_history_restart = sched_file_get_num_restart_files( ecl_config->sched_file ) - 1;   /* We keep track of this - so we can stop assimilation at the end of history */
  }
  
  if (config_has_set_item(config , "INIT_SECTION")) {

  /* The semantic regarding INIT_SECTION is as follows:
  
       1. If the INIT_SECTION points to an existing file - the
          ecl_config->equil_init_file is set to the absolute path of
          this file.

       2. If the INIT_SECTION points to a not existing file:

          a. We assert that INIT_SECTION points to a pure filename,
             i.e. /some/path/which/does/not/exist is NOT accepted.
          b. The ecl_config->equil_init_file is set to point to this
             file.
          c. WE TRUST THE USER TO SUPPLY CONTENT (THROUGH SOME FUNKY
             FORWARD MODEL) IN THE RUNPATH. This can unfortunately not
             be checked/verified before the ECLIPSE simulation fails.
    */
    
    const char * init_section = config_iget(config , "INIT_SECTION" , 0,0);
    if (util_file_exists( init_section ))
      ecl_config->equil_init_file = util_alloc_realpath(init_section);
    else {
      char * filename;
      char * basename;
      char * extension;
      
      util_alloc_file_components( init_section , NULL , &basename , &extension);
      filename = util_alloc_filename( NULL , basename , extension);
      if (strcmp( filename , init_section) == 0) 
	ecl_config->equil_init_file = filename;
      else {
	util_abort("%s: When INIT_SECTION:%s is set to a non-existing file - you can not have any path components.\n",__func__ , init_section);
	util_safe_free( filename );
      }
      
      util_safe_free( basename );
      util_safe_free( extension );
    }
  } else
    ecl_config->can_restart = false;
  /*
      The user has not supplied a INIT_SECTION keyword whatsoever, 
      this essentially meens that we can not restart - because:

      1. The EQUIL section must be inlined in the DATAFILE without any
         special markup.
      
      2. ECLIPSE will fail hard if the datafile contains both an EQUIL
         section and a restart statement, and when we have not marked
         the EQUIL section specially with the INIT_SECTION keyword it
         is impossible for ERT to dynamically change between a
         datafile with initialisation and a datafile for restart.
         
      IFF the user has no intentitions of any form of restart, this is
      perfectly legitemate.
  */

  if (config_item_set(config , "GRID"))
    ecl_config->grid = ecl_grid_alloc( config_iget(config , "GRID" , 0,0) );
  else
    ecl_config->grid = NULL;

  return ecl_config;
}


void ecl_config_free(ecl_config_type * ecl_config) {
  ecl_io_config_free( ecl_config->io_config );
  path_fmt_free( ecl_config->eclbase );
  set_free( ecl_config->static_kw_set );
  free(ecl_config->data_file);
  sched_file_free(ecl_config->sched_file);
  free(ecl_config->schedule_target_file);

  util_safe_free(ecl_config->equil_init_file);

  if (ecl_config->grid != NULL)
    ecl_grid_free( ecl_config->grid );
  
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

bool ecl_config_get_formatted(const ecl_config_type * ecl_config)        { return ecl_io_config_get_formatted(ecl_config->io_config); }
bool ecl_config_get_unified_restart(const ecl_config_type * ecl_config)  { return ecl_io_config_get_unified_restart( ecl_config->io_config ); }
bool ecl_config_get_unified_summary(const ecl_config_type * ecl_config)  { return ecl_io_config_get_unified_summary( ecl_config->io_config ); }

