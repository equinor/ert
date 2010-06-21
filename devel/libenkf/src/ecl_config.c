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
#include <ecl_sum.h>
#include <sched_file.h>
#include <config.h>
#include <ecl_config.h>
#include <parser.h>
#include "config_keys.h"
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
  stringlist_type    * user_static_kw;
  char               * data_file;              	   /* Eclipse data file. */
  time_t               start_date;                 /* The start date of the ECLIPSE simulation - parsed from the data_file. */
  ecl_sum_type       * refcase;                    /* Refcase - can be NULL. */
  ecl_grid_type      * grid;                   	   /* The grid which is active for this model. */
  char               * schedule_prediction_file;   /* Name of schedule prediction file - observe that this is internally handled as a gen_kw node. */
  char               * schedule_target_file;   	   /* File name to write schedule info to */
  char               * input_init_section;         /* File name for ECLIPSE (EQUIL) initialisation - can be NULL if the user has not supplied INIT_SECTION. */
  char               * init_section;               /* Equal to the full path of input_init_section IFF input_init_section points to an existing file - otherwise equal to input_init_section. */
  int                  last_history_restart;
  bool                 can_restart;                /* Have we found the <INIT> tag in the data file? */
  int                  num_cpu;                    /* We should parse the ECLIPSE data file and determine how many cpus this eclipse file needs. */
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
  ecl_config->start_date = ecl_util_get_start_date( ecl_config->data_file );
  ecl_config->num_cpu    = ecl_util_get_num_cpu( ecl_config->data_file );
  printf("oK - need:%d cpu \n",ecl_config->num_cpu);
}


const char * ecl_config_get_data_file(const ecl_config_type * ecl_config) {
  return ecl_config->data_file;
}


int ecl_config_get_num_cpu( const ecl_config_type * ecl_config ) {
  return ecl_config->num_cpu;
}

const char * ecl_config_get_schedule_prediction_file( const ecl_config_type * ecl_config ) {
  return ecl_config->schedule_prediction_file;
}

/**
   Observe: The real schedule prediction functionality is implemented
   as a special GEN_KW node in ensemble_config.
*/


void ecl_config_set_schedule_prediction_file( ecl_config_type * ecl_config , const char * schedule_prediction_file ) {
  ecl_config->schedule_prediction_file = util_realloc_string_copy( ecl_config->schedule_prediction_file , schedule_prediction_file );
}


const char * ecl_config_get_schedule_file( const ecl_config_type * ecl_config ) {
  if (ecl_config->sched_file != NULL) 
    return sched_file_iget_filename( ecl_config->sched_file , 0 );
  else 
    return NULL;
}


/**
   Observe: This function makes a hard assumption that the
   ecl_config->start_date has already been set. (And should not be
   changed either ...)
*/

void ecl_config_set_schedule_file( ecl_config_type * ecl_config , const char * schedule_file ) {
  if (ecl_config->start_date == -1)
    util_abort("%s: must set ecl_data_file first \n",__func__);
  {
    char * base;  /* The schedule target file will be without any path component */
    char * ext;
    util_alloc_file_components(schedule_file , NULL , &base , &ext);
    ecl_config->schedule_target_file = util_alloc_filename(NULL , base , ext);
    free(ext);
    free(base);
  }
  ecl_config->sched_file = sched_file_alloc( ecl_config->start_date );

  /* Fixed length schedule_kw  - not supported at the moment */
  //{
  //  if (config_has_set_item( config , "ADD_FIXED_LENGTH_SCHEDULE_KW")) {
  //    int iocc;
  //    for (iocc = 0; iocc < config_get_occurences(config , "ADD_FIXED_LENGTH_SCHEDULE_KW"); iocc++) 
  //      sched_file_add_fixed_length_kw( ecl_config->sched_file , 
  //                                      config_iget(config , "ADD_FIXED_LENGTH_SCHEDULE_KW" , iocc , 0) , 
  //                                      config_iget_as_int(config , "ADD_FIXED_LENGTH_SCHEDULE_KW" , iocc , 1));
  //    
  //  }
  //}
  
  sched_file_parse(ecl_config->sched_file , schedule_file );
  ecl_config->last_history_restart = sched_file_get_num_restart_files( ecl_config->sched_file ) - 1;   /* We keep track of this - so we can stop assimilation at the end of history */
}



void ecl_config_set_eclbase( ecl_config_type * ecl_config , const char * eclbase_fmt ) {
  if (ecl_config->eclbase != NULL)
    path_fmt_free( ecl_config->eclbase );
  ecl_config->eclbase = path_fmt_alloc_path_fmt( eclbase_fmt );
}


/**
   Observe that this function returns a (char *) - corresponding to
   the argument used when calling the ecl_config_set_eclbase()
   function, and not a path_fmt instance.
*/
 
const char * ecl_config_get_eclbase( const ecl_config_type * ecl_config ) {
  return path_fmt_get_fmt( ecl_config->eclbase );
}


/**
   Can be called with @refcase == NULL - which amounts to clearing the
   current refcase. 
*/
void ecl_config_load_refcase( ecl_config_type * ecl_config , const char * refcase ){ 
  if (ecl_config->refcase != NULL) {
    if (refcase == NULL) {    /* Clear the refcase */
      ecl_sum_free( ecl_config->refcase );
      ecl_config->refcase = NULL;
    } else {                  /* Check if the currently loaded case is the same as refcase */
      if (!ecl_sum_same_case( ecl_config->refcase , refcase )) {
        ecl_sum_free( ecl_config->refcase );
        ecl_config->refcase = ecl_sum_fread_alloc_case( refcase , DEFAULT_SUMMARY_JOIN );
      }
    }
  } else {
    if (refcase != NULL)
      ecl_config->refcase = ecl_sum_fread_alloc_case( refcase , DEFAULT_SUMMARY_JOIN );
  }
}


/**
   Will return NULL if no refcase is set.
*/
const char * ecl_config_get_refcase_name( const ecl_config_type * ecl_config) {

  if (ecl_config->refcase == NULL)
    return NULL;
  else
    return ecl_sum_get_case( ecl_config->refcase );

}

/**
   This function will clear the list of static keywords supplied by
   the user. The default built in keywords are not touched.
*/

void ecl_config_clear_static_kw( ecl_config_type * ecl_config ) {
  ecl_config->include_all_static_kw = false;
  stringlist_clear( ecl_config->user_static_kw );
}

/**
   Returns a stringlist of the user-defined static keywords.
*/
stringlist_type * ecl_config_get_static_kw_list( const ecl_config_type * ecl_config ) {
  return ecl_config->user_static_kw;
}


void ecl_config_set_init_section( ecl_config_type * ecl_config , const char * input_init_section ) {
  /* The semantic regarding INIT_SECTION is as follows:
  
       1. If the INIT_SECTION points to an existing file - the
          ecl_config->input_init_section is set to the absolute path of
          this file.

       2. If the INIT_SECTION points to a not existing file:

          a. We assert that INIT_SECTION points to a pure filename,
             i.e. /some/path/which/does/not/exist is NOT accepted.
          b. The ecl_config->input_init_section is set to point to this
             file.
          c. WE TRUST THE USER TO SUPPLY CONTENT (THROUGH SOME FUNKY
             FORWARD MODEL) IN THE RUNPATH. This can unfortunately not
             be checked/verified before the ECLIPSE simulation fails.


     The INIT_SECTION keyword and <INIT> in the datafile (checked with
     ecl_config->can_restart) interplay as follows: 


     CASE   |   INIT_SECTION  |  <INIT>    | OK ?
     ---------------------------------------------
     0      |   Present       | Present    |  Yes 
     1      |   Not Present   | Present    |  No    
     2      |   Present       | Not present|  No
     3      |   Not Present   | Not present|  Yes
     ---------------------------------------------


     Case 0: This is the most flexible case, which can do arbitrary
        restart.

     Case 1: In this case the datafile will contain a <INIT> tag, we
        we do not have the info to replace that tag with for
        initialisation, and ECLIPSE will fail. Strictly speaking this
        case can actually restart, but that is not enough - we let
        this case fail hard.

     Case 2: We have some INIT_SECTION infor, but no tag in he
        datafile to update. If the datafile has embedded
        initialisation info this case will work for init; but it is
        logically flawed, and not accepted. Currently only a warning.

     Case 3: This case has just the right amount of information for
        initialisation, but it is 'consistently unable' to restart.
     
  */
  if (ecl_config->can_restart) {  /* The <INIT> tag is set. */
    ecl_config->input_init_section = util_realloc_string_copy( ecl_config->input_init_section , input_init_section );  /* input_init_section = path/to/init_section         */
    if (util_file_exists( ecl_config->input_init_section )) {                                                           /* init_section       = $CWD/path/to/init_section */ 
      util_safe_free( ecl_config->init_section );
      ecl_config->init_section = util_alloc_realpath(input_init_section);
    } else {
      char * path;
      
      util_alloc_file_components( ecl_config->input_init_section , &path , NULL , NULL );
      if (path != NULL) 
        util_abort("%s: When INIT_SECTION:%s is set to a non-existing file - you can not have any path components.\n",__func__ , input_init_section);
      
      util_safe_free( path );
      ecl_config->init_section = util_alloc_string_copy(input_init_section);
    }
  } else
    /* 
       The <INIT> tag is not set - we can not utilize the
       input_init_section info, and we just ignore it.
    */
    fprintf(stderr,"** Warning: <INIT> tag was not found in datafile - can not utilize INIT_SECTION keyword - ignored.\n");
}



static void ecl_config_init_static_kw( ecl_config_type * ecl_config ) {
  int i;
  for (i=0; i < NUM_STATIC_KW; i++)
    set_add_key( ecl_config->static_kw_set , DEFAULT_STATIC_KW[i]);
}



ecl_config_type * ecl_config_alloc( const config_type * config ) {
  ecl_config_type * ecl_config         = util_malloc(sizeof * ecl_config , __func__);
  ecl_config->io_config 	       = ecl_io_config_alloc( DEFAULT_FORMATTED , DEFAULT_UNIFIED , DEFAULT_UNIFIED );
  ecl_config->eclbase                  = NULL;
  ecl_config->include_all_static_kw    = false;
  ecl_config->static_kw_set            = set_alloc_empty();
  ecl_config->user_static_kw           = stringlist_alloc_new();
  ecl_config->num_cpu                  = -1;
  ecl_config->data_file                = NULL;
  ecl_config->input_init_section       = NULL; 
  ecl_config->init_section             = NULL;
  ecl_config->refcase                  = NULL;
  ecl_config->grid                     = NULL;
  ecl_config->can_restart              = false;
  ecl_config->start_date               = -1;
  ecl_config->sched_file               = NULL;
  ecl_config->schedule_prediction_file = NULL;

  /*****************************************************************/
  ecl_config_init_static_kw( ecl_config );
  ecl_config_set_eclbase( ecl_config , config_iget(config , "ECLBASE" ,0,0) );
  ecl_config_set_data_file( ecl_config , config_iget( config , DATA_FILE_KEY ,0,0));
  ecl_config_set_schedule_file( ecl_config , config_iget( config , SCHEDULE_FILE_KEY ,0,0));
  ecl_config_clear_static_kw( ecl_config );
  if (config_item_set(config , "GRID"))
    ecl_config_set_grid( ecl_config , config_iget(config , "GRID" , 0,0) );
  
  if (config_item_set( config , "REFCASE")) 
    ecl_config_load_refcase( ecl_config , config_get_value( config , "REFCASE" ));

  if (config_has_set_item(config , "INIT_SECTION")) 
    ecl_config_set_init_section( ecl_config , config_get_value( config , "INIT_SECTION" ));
  else 
    if (ecl_config->can_restart) 
      /** 
          This is a hard error - the datafile contains <INIT>, however
          the config file does NOT contain INIT_SECTION, i.e. we have
          no information to fill in for the <INIT> section. This case
          will not be able to initialize an ECLIPSE model, and that is
          broken behaviour. 
      */
      util_exit("Sorry: when the datafile contains <INIT> the config file MUST have the INIT_SECTIOn keyword. \n");
  
  /*
      The user has not supplied a INIT_SECTION keyword whatsoever, 
      this essentially means that we can not restart - because:

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
  
  return ecl_config;
}



void ecl_config_free(ecl_config_type * ecl_config) {
  ecl_io_config_free( ecl_config->io_config );
  path_fmt_free( ecl_config->eclbase );
  set_free( ecl_config->static_kw_set );
  stringlist_free( ecl_config->user_static_kw );
  free(ecl_config->data_file);
  sched_file_free(ecl_config->sched_file);
  free(ecl_config->schedule_target_file);

  util_safe_free(ecl_config->input_init_section);
  util_safe_free(ecl_config->init_section);
  util_safe_free(ecl_config->schedule_prediction_file);

  if (ecl_config->grid != NULL)
    ecl_grid_free( ecl_config->grid );

  if (ecl_config->refcase != NULL)
    ecl_sum_free( ecl_config->refcase );
  
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
    if (!stringlist_contains( ecl_config->user_static_kw , kw ))
      stringlist_append_owned_ref( ecl_config->user_static_kw , kw );
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
  else {
    if (set_has_key(ecl_config->static_kw_set , kw))
      return true;
    else
      return stringlist_contains( ecl_config->user_static_kw , kw );
  }
}


  

ecl_grid_type * ecl_config_get_grid(const ecl_config_type * ecl_config) {
  return ecl_config->grid;
}


const char * ecl_config_get_gridfile( const ecl_config_type * ecl_config ) {
  return ecl_grid_get_name( ecl_config->grid );
}


/**
   The ecl_config object isolated supports run-time changing of the
   grid, however this does not (in general) apply to the system as a
   whole. Other objects which internalize pointers (i.e. field_config
   objects) to an ecl_grid_type instance will be left with dangling
   pointers; and things will probably die an ugly death. So - changing
   grid runtime should be done with extreme care.
*/

void ecl_config_set_grid( ecl_config_type * ecl_config , const char * grid_file ) {
  if (ecl_config->grid != NULL)
    ecl_grid_free( ecl_config->grid );
  ecl_config->grid = ecl_grid_alloc( grid_file );
}


const ecl_sum_type * ecl_config_get_refcase(const ecl_config_type * ecl_config) {
  return ecl_config->refcase;
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


/**
   This just returns the string which has been set with the
   ecl_config_set_init_section() function, whereas the
   ecl_config_get_equil_init_file() function will return the absolute
   path to the init_section (if it exists).
*/


const char * ecl_config_get_init_section(const ecl_config_type * ecl_config) {
  return ecl_config->input_init_section;
}

const char * ecl_config_get_equil_init_file(const ecl_config_type * ecl_config) {
  return ecl_config->init_section;
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


void ecl_config_fprintf_config( const ecl_config_type * ecl_config , FILE * stream ) {
  fprintf( stream , CONFIG_KEY_FORMAT      , DATA_FILE_KEY );
  fprintf( stream , CONFIG_ENDVALUE_FORMAT , ecl_config->data_file );
  
  fprintf( stream , CONFIG_KEY_FORMAT      , SCHEDULE_FILE_KEY );
  fprintf( stream , CONFIG_ENDVALUE_FORMAT , sched_file_iget_filename( ecl_config->sched_file , 0));
}
