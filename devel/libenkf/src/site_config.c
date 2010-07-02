#include <stdlib.h>
#include <string.h>
#include <job_queue.h>
#include <util.h>
#include <ext_job.h>
#include <ext_joblist.h>
#include <stringlist.h>
#include <config.h>
#include <site_config.h>
#include <local_driver.h>
#include <lsf_driver.h>
#include <lsf_request.h>
#include <rsh_driver.h>
#include <basic_queue_driver.h>
#include <sys/types.h>
#include <vector.h>
#include <sys/stat.h>
#include "enkf_defaults.h"
#include "config_keys.h"

/**
   This struct contains information which is specific to the site
   where this enkf instance is running. Pointers to the fields in this
   structure are passed on to e.g. the enkf_state->shared_info object,
   but this struct is the *OWNER* of this information, and hence
   responsible for booting and deleting these objects. 

   The settings held by the site_config object are by default set in
   the site-wide configuration file, but they can also be overridden
   in the users configuration file. This makes both parsing,
   validating and also storing the configuration information a bit
   more tricky:

   Parsing:
   --------
   When parsing the user configuration file all settings are optional,
   that means that the required validation of the config system, can
   not be used, instead every get must be preceeded by: 

      if (config_item_set(config , KEY)) ...

   Furthermore everything is done twice; first with config as a
   site-config instance, and later as user-config instance.


   Saving:
   -------
   A setting which originates from the site_config file should not be
   stored in the user's config file, but additions/overrides from the
   user's config file should of course be saved. This is solved as
   follows:

     1. When parsing the site configuration file the field
        'user_parse' is set to false; before parsing the user config
        file this must be set to true with the function
        site_config_init_user_parse().

     2. The jobs in the joblist have an internal 'private_job' flag
        which is set according to the 'user_parse' flag of the
        site_config structure.
        
     3. For all other fields the xxxx_set() actions are stored as
        string items in the vector 'user_action'; and when saving the
        configuration this vector is played. (This of course can lead
        to duplicated entries ... - but that should be OK??)


*/

struct site_config_struct {
  ext_joblist_type 	* joblist;                /* The list of external jobs which have been installed. 
                   	                             These jobs will be the parts of the forward model. */
  hash_type             * env_variables;
  hash_type             * initial_variables;      /* Need to store the initial values so we can roll back. */

  hash_type             * initial_path_variables;
  stringlist_type       * path_variables;         /* We can update the same path variable several times - i.e. it can not be a hash table. */
  stringlist_type       * path_values;

  char                  * license_root_path;      /* The license_root_path value set by the user. */
  char                  * __license_root_path;    /* The license_root_path value actually used - includes a user/pid subdirectory. */
  mode_t                  umask;

  int                     max_running_lsf;        /* Need to hold the detailed information about the         */
  char                  * lsf_queue_name;         /* various drivers here to be able to "hot-switch" driver. */
  char                  * lsf_request;  
  
  int                     max_running_rsh;
  hash_type             * rsh_host_list;
  char                  * rsh_command;

  int                     max_running_local;

  job_driver_type         driver_type;
  int                     max_submit;             
  char                  * job_script;            


  int                     num_cpu;                /* The number of cpu's used to run the forward model - currently only relevant for ECLIPSE and LSF; read automatically from the ECLIPSE data file. */
  job_queue_type   	* job_queue;              /* The queue instance which will run the external jobs. */

  bool                    user_parse;
  vector_type           * user_action;
};


/**
   Uses variable length argument list - MUST terminate with NULL. All
   arguments MUST be of type (char *).
   
   Example:
   site_config_user_store( site_config , RSH_HOST_LIST_KEY , "HOST1:2", "HOST3:1" , "OTHER_HOST:8" , NULL);
*/

static void site_config_user_store( site_config_type * site_config , const char * key , ...) {
  va_list ap;
  va_start(ap , key);
  {
    char * arg;
    stringlist_type * args = stringlist_alloc_new();
    stringlist_append_copy( args , key );
    do {
      arg = va_arg( ap , char *);
      if (arg != NULL)
        stringlist_append_copy( args , arg);
    } while (arg != NULL);
    vector_append_owned_ref( site_config->user_action , args , stringlist_free__);
  }
  va_end( ap );
}


void site_config_set_num_cpu( site_config_type * site_config , int num_cpu ) {
  job_queue_set_num_cpu( site_config->job_queue , num_cpu );
}


void site_config_set_umask( site_config_type * site_config , mode_t new_mask) {
  site_config->umask = new_mask;
  umask( new_mask );
  if (site_config->user_parse) {
    char * umask_string = util_alloc_sprintf("%o" , new_mask);
    site_config_user_store( site_config , UMASK_KEY , umask_string, NULL);
    free( umask_string );
  }
}

mode_t site_config_get_umask( const site_config_type * site_config ) {
  return site_config->umask;
}


/**
   This site_config object is not really ready for prime time.
*/
site_config_type * site_config_alloc_empty() {
  site_config_type * site_config = util_malloc( sizeof * site_config , __func__);
  
  site_config->joblist                = ext_joblist_alloc( );
  site_config->job_queue              = NULL;

  site_config->lsf_queue_name         = NULL;
  site_config->lsf_request            = NULL;
  site_config->rsh_host_list          = hash_alloc();
  site_config->rsh_command            = NULL;
  site_config->license_root_path      = NULL;
  site_config->__license_root_path    = NULL;
  site_config->job_script             = NULL;  

  site_config->user_parse             = false;
  site_config->num_cpu                = 0;
  site_config->max_running_local      = 0;
  site_config->max_running_lsf        = 0;
  site_config->max_running_rsh        = 0;
  site_config->user_action            = vector_alloc_new();
  site_config->driver_type            = NULL_DRIVER;

  /* Some hooops to get the current umask. */ 
  site_config->umask                  = umask( 0 );
  site_config_set_umask( site_config , site_config->umask );

  site_config->initial_variables      = hash_alloc();
  site_config->env_variables          = hash_alloc();
  site_config->path_variables         = stringlist_alloc_new();
  site_config->path_values            = stringlist_alloc_new();
  site_config->initial_path_variables = hash_alloc();
  
  site_config_set_max_submit( site_config , DEFAULT_MAX_SUBMIT );
  return site_config;
}

const char * site_config_get_license_root_path__( const site_config_type * site_config ) {
  return site_config->__license_root_path;
}


/**
   Observe that this variable can not "really" be set to different
   values during a simulation, when creating ext_job instances they
   will store a pointer to this variable on creation, if the variable
   is later changed they will be left with a dangling copy. That is
   not particularly elegant, however it should nonetheless work.
*/

void site_config_set_license_root_path( site_config_type * site_config , const char * license_root_path) {
  /**
    Appending /user/pid to the license root path. Everything
    including the pid is removed when exiting (gracefully ...).
    
    Dangling license directories after a crash can just be removed.
  */
  
  site_config->license_root_path   = util_realloc_string_copy( site_config->license_root_path , license_root_path );
  site_config->__license_root_path = util_realloc_sprintf(site_config->__license_root_path , "%s%c%s%c%d" , license_root_path , UTIL_PATH_SEP_CHAR , getenv("USER") , UTIL_PATH_SEP_CHAR , getpid());

  if (site_config->user_parse)
    site_config_user_store( site_config , LICENSE_PATH_KEY , license_root_path , NULL );
}



void site_config_init_user_parse( site_config_type * site_config ) {
  site_config->user_parse = true;
}


/**
   Will return 0 if the job is added correctly, and a non-zero (not
   documented ...) error code if the job is not added. 
*/

int site_config_install_job(site_config_type * site_config , const char * job_name , const char * install_file) {
  ext_job_type * new_job = ext_job_fscanf_alloc(job_name , site_config->__license_root_path , site_config->user_parse , install_file) ;
  if (new_job != NULL) {
    ext_joblist_add_job(site_config->joblist , job_name , new_job);
    return 0;
  } else 
    return 1; /* Some undocumented error condition - the job is NOT added. */
}


/**
   Will NOT remove shared jobs.
*/
bool site_config_del_job( site_config_type * site_config , const char * job_name ) {
  return ext_joblist_del_job( site_config->joblist , job_name );
}



static void site_config_add_jobs(site_config_type * site_config , const config_type * config) {
  int  i;
  
  stringlist_type *item_list = config_alloc_complete_stringlist(config , INSTALL_JOB_KEY);

  for (i=0; i < stringlist_get_size(item_list); i+=2) 
    site_config_install_job(site_config , stringlist_iget(item_list , i) , stringlist_iget(item_list , i + 1));

  stringlist_free(item_list);
}


hash_type * site_config_get_env_hash( const site_config_type * site_config ) {
  return site_config->env_variables;
}

stringlist_type * site_config_get_path_variables( const site_config_type * site_config ) {
  return site_config->path_variables;
}

stringlist_type * site_config_get_path_values( const site_config_type * site_config ) {
  return site_config->path_values;
}


void site_config_setenv( site_config_type * site_config , const char * variable, const char * value) {
  /* Store the current variable, to be able to roll back */
  if (!hash_has_key( site_config->initial_variables , variable )) 
    hash_insert_hash_owned_ref( site_config->initial_variables , variable , util_alloc_string_copy( getenv( variable )) , util_safe_free); 
  
  hash_insert_hash_owned_ref( site_config->env_variables , variable , util_alloc_string_copy( value ) , free);
  setenv( variable , value , 1);  /* This will update the environment if the variable has already been set to another value. */
  if (site_config->user_parse)
    site_config_user_store( site_config , SETENV_KEY , variable , value , NULL);
}


void site_config_clear_env( site_config_type * site_config ) {
  hash_clear( site_config->env_variables );
  {
    /* Recover the original values. */
    hash_iter_type * hash_iter = hash_iter_alloc( site_config->initial_variables );
    while (!hash_iter_is_complete( hash_iter )) {
      const char * var       = hash_iter_get_next_key( hash_iter );
      const char * old_value = hash_get( site_config->initial_variables , var );

      if (old_value == NULL)
        unsetenv( var );
      else
        setenv( var , old_value , 1 );
    }
  }
}


void site_config_clear_pathvar( site_config_type * site_config ) {
  stringlist_clear( site_config->path_variables );
  stringlist_clear( site_config->path_values );
  {
    /* Recover the original values. */
    hash_iter_type * hash_iter = hash_iter_alloc( site_config->initial_path_variables );
    while (!hash_iter_is_complete( hash_iter )) {
      const char * var       = hash_iter_get_next_key( hash_iter );
      const char * old_value = hash_get( site_config->initial_path_variables , var );

      if (old_value == NULL)
        unsetenv( var );
      else
        setenv( var , old_value , 1 );
    }
  }
}


void site_config_update_pathvar( site_config_type * site_config , const char * pathvar , const char * value) {
  /* Store the current variable, to be able to roll back */
  if (!hash_has_key( site_config->initial_path_variables , pathvar )) 
    hash_insert_hash_owned_ref( site_config->initial_path_variables , pathvar , util_alloc_string_copy( getenv( pathvar )) , util_safe_free); 
    
  stringlist_append_copy( site_config->path_variables , pathvar );
  stringlist_append_copy( site_config->path_values    , value   );
  
  util_update_path_var( pathvar , value , false );
  if (site_config->user_parse)
    site_config_user_store( site_config , UPDATE_PATH_KEY , pathvar , value , NULL);
}



/**
   These functions can be called repeatedly if you should want to
   change driver characteristics run-time. The job_queue will discard
   the old driver, if a driver is already installed. 
*/
static void site_config_install_LOCAL_job_queue(site_config_type * site_config ) {
  basic_queue_driver_type * driver = local_driver_alloc();
  job_queue_set_driver( site_config->job_queue , driver );
  job_queue_set_max_running( site_config->job_queue , site_config->max_running_local );
}


static void site_config_install_RSH_job_queue(site_config_type * site_config) {
  basic_queue_driver_type * driver = rsh_driver_alloc(site_config->rsh_command , site_config->rsh_host_list);
  job_queue_set_driver( site_config->job_queue , driver );
  job_queue_set_max_running( site_config->job_queue , site_config->max_running_rsh );
}


static void site_config_install_LSF_job_queue(site_config_type * site_config ) {
  basic_queue_driver_type * driver = lsf_driver_alloc( site_config->lsf_queue_name , site_config->num_cpu);
  job_queue_set_driver( site_config->job_queue , driver );
  job_queue_set_max_running( site_config->job_queue , site_config->max_running_lsf );
}


/*****************************************************************/
/**
   This is quite awkward because the max_running variable is located
   both in the job_queue instance, and in each separate driver
   individually:

    o If you call set_max_running - i.e. without specifiying a
      particular driver, it will tell the queue system to use this
      many jobs, and also look up the currently active driver and
      update the internal info on that.

    o If you tell a specific driver a value for max_running, it will
      update the internal field for that driver, AND the queue IFF the
      queue is currently running this driver; otherwise the queue will
      be left untouched.

   What a mess.   
*/


void site_config_set_max_running( site_config_type * site_config , int max_running ) {
  job_driver_type current_driver = job_queue_get_driver_type( site_config->job_queue );
  switch(current_driver) {
  case( NULL_DRIVER ):
    util_abort("%s: trying to set the number of jobs before a driver has been allocated \n",__func__);
    break;
  case( LSF_DRIVER ):
    site_config_set_max_running_lsf( site_config , max_running );
    break;
  case( RSH_DRIVER ):
    site_config_set_max_running_rsh( site_config , max_running );
    break;
  case( LOCAL_DRIVER ):
    site_config_set_max_running_local( site_config , max_running );
    break;
  default:
    util_abort("%s: what the fuck? \n",__func__);
  }
}


void site_config_set_max_running_lsf( site_config_type * site_config , int max_running_lsf) {
  site_config->max_running_lsf = max_running_lsf;
  if (site_config->job_queue != NULL) {
    if (job_queue_get_driver_type( site_config->job_queue ) == LSF_DRIVER)
      job_queue_set_max_running( site_config->job_queue , max_running_lsf );
  }

  if (site_config->user_parse) {
    char * max_running_lsf_string = util_alloc_sprintf( "%d" , max_running_lsf );
    site_config_user_store( site_config , MAX_RUNNING_LSF_KEY , max_running_lsf_string , NULL );
    free( max_running_lsf_string );
  }
}

int site_config_get_max_running_lsf( const site_config_type * site_config ) {
  return site_config->max_running_lsf;
}

void site_config_set_max_running_rsh( site_config_type * site_config , int max_running_rsh) {
  site_config->max_running_rsh = max_running_rsh;
  if (site_config->job_queue != NULL) {
    if (job_queue_get_driver_type( site_config->job_queue ) == RSH_DRIVER)
      job_queue_set_max_running( site_config->job_queue , max_running_rsh );
  }
  if (site_config->user_parse) {
    char * max_running_rsh_string = util_alloc_sprintf( "%d" , max_running_rsh );
    site_config_user_store( site_config , MAX_RUNNING_RSH_KEY , max_running_rsh_string , NULL );
    free( max_running_rsh_string );
  }
}

int site_config_get_max_running_rsh( const site_config_type * site_config) {
  return site_config->max_running_rsh;
}

void site_config_set_max_running_local( site_config_type * site_config , int max_running_local) {
  site_config->max_running_local = max_running_local;
  if (site_config->job_queue != NULL) {
    if (job_queue_get_driver_type( site_config->job_queue ) == LOCAL_DRIVER)
      job_queue_set_max_running( site_config->job_queue , max_running_local );
  }
  if (site_config->user_parse) {
    char * max_running_local_string = util_alloc_sprintf( "%d" , max_running_local );
    site_config_user_store( site_config , MAX_RUNNING_LOCAL_KEY , max_running_local_string , NULL );
    free( max_running_local_string );
  }
}

int site_config_get_max_running_local( const site_config_type * site_config ) {
  return site_config->max_running_local;
}

/*****************************************************************/


/*****************************************************************/

void site_config_clear_rsh_host_list( site_config_type * site_config ) {
  hash_clear( site_config->rsh_host_list );
}


hash_type * site_config_get_rsh_host_list( const site_config_type * site_config ) {
  return site_config->rsh_host_list;
}


void site_config_add_rsh_host( site_config_type * site_config , const char * rsh_host , int max_running ) {
  hash_insert_int( site_config->rsh_host_list , rsh_host , max_running );

  if (site_config->user_parse) {
    char * new_host = util_alloc_sprintf( "%s:%d" , rsh_host , max_running);
    site_config_user_store( site_config , RSH_HOST_LIST_KEY , new_host , NULL);
    free( new_host);
  }
}


/*****************************************************************/

void site_config_set_lsf_queue( site_config_type * site_config , const char * lsf_queue) {
  site_config->lsf_queue_name = util_realloc_string_copy( site_config->lsf_queue_name , lsf_queue);
  if (site_config->user_parse) 
    site_config_user_store( site_config , LSF_QUEUE_KEY , lsf_queue , NULL);
}


const char * site_config_get_lsf_queue( const site_config_type * site_config ) {
  return site_config->lsf_queue_name;
}


void site_config_set_lsf_request( site_config_type * site_config , const char * lsf_request) {
  site_config->lsf_request = util_realloc_string_copy( site_config->lsf_request , lsf_request);
  if (site_config->user_parse) 
    site_config_user_store( site_config , LSF_RESOURCES_KEY , lsf_request , NULL);
}

const char * site_config_get_lsf_request( const site_config_type * site_config ) {
  return site_config->lsf_request;
}

/*****************************************************************/

void site_config_set_rsh_command( site_config_type * site_config , const char * rsh_command) {
  site_config->rsh_command = util_realloc_string_copy( site_config->rsh_command  , rsh_command);
  if (site_config->user_parse) 
    site_config_user_store( site_config , RSH_COMMAND_KEY , rsh_command , NULL);
}

const char * site_config_get_rsh_command( const site_config_type * site_config ) {
  return site_config->rsh_command;
}





/*****************************************************************/


const char * site_config_get_queue_name( const site_config_type * site_config ) {
  return job_queue_get_driver_name( site_config->job_queue );
}


static void site_config_set_job_queue__( site_config_type * site_config , job_driver_type driver_type) {
  site_config->driver_type = driver_type;
  if (site_config->job_queue != NULL) {
    switch (driver_type) {
    case(LSF_DRIVER):
      site_config_install_LSF_job_queue( site_config );
      break;
    case(RSH_DRIVER):
      site_config_install_RSH_job_queue( site_config );
      break;
    case(LOCAL_DRIVER):
      site_config_install_LOCAL_job_queue( site_config );
      break;
    default:
      util_abort("%s: internal error \n",__func__);
    }
  }
}


void site_config_set_job_queue( site_config_type * site_config , const char * queue_name ) {
  job_driver_type driver_type = job_queue_lookup_driver_name( queue_name );
  site_config_set_job_queue__( site_config , driver_type );
}

bool site_config_queue_is_running( const site_config_type * site_config ) {
  return job_queue_is_running( site_config->job_queue );
}


void site_config_set_job_script( site_config_type * site_config , const char * job_script ) {
  site_config->job_script = util_realloc_string_copy( site_config->job_script , job_script );
  if (site_config->job_queue != NULL)
    job_queue_set_run_cmd( site_config->job_queue  , job_script );

  if (site_config->user_parse) 
    site_config_user_store( site_config , JOB_SCRIPT_KEY , job_script , NULL);
}


const char * site_config_get_job_script( const site_config_type * site_config ) {
  return job_queue_get_run_cmd( site_config->job_queue );
}


void site_config_set_max_submit( site_config_type * site_config , int max_submit ) {
  site_config->max_submit = max_submit;
  if (site_config->job_queue != NULL)
    job_queue_set_max_submit(site_config->job_queue , max_submit);
}


int site_config_get_max_submit(const site_config_type * site_config ) {
  return job_queue_get_max_submit(site_config->job_queue);
}


static void site_config_install_job_queue(site_config_type  * site_config ) {
  if (site_config->job_script == NULL)
    util_exit("Must set the path to the job script with the %s key in the site_config / config file\n",JOB_SCRIPT_KEY);
  
  site_config->job_queue = job_queue_alloc(0 , 0 , site_config->max_submit , site_config->job_script);

  /* 
     All the various driver options are set, unconditionally of which
     driver is actually selected in the end.
  */
  site_config_set_job_queue__( site_config , site_config->driver_type );
}



/**
   This function will be called twice, first when the config instance
   is an internalization of the site-wide configuration file, and
   secondly when config is an internalisation of the user's
   configuration file. The @user_config parameter will be true in the
   latter case.
*/


void site_config_init(site_config_type * site_config , const config_type * config, bool user_config) {
  site_config_add_jobs(site_config , config);
  {
    int i;
    for (i = 0; i < config_get_occurences( config , SETENV_KEY); i++) {
      const stringlist_type * tokens = config_iget_stringlist_ref(config , SETENV_KEY , i);
      const char * var               = stringlist_iget( tokens , 0);
      const char * value             = stringlist_iget( tokens , 1);

      site_config_setenv( site_config , var , value );
    }
    
    for (i=0; i < config_get_occurences( config, UPDATE_PATH_KEY); i++) {
      const stringlist_type * tokens = config_iget_stringlist_ref(config , UPDATE_PATH_KEY , i);
      const char * path              = stringlist_iget( tokens , 0);
      const char * value             = stringlist_iget( tokens , 1);
      
      site_config_update_pathvar( site_config , path , value );
    }
  }
  
  /* 
     Set the umask for all file creation. A value of '0' will ensure
     that all files and directories are created with 'equal rights'
     for everyone - might be handy if you are helping someone... The
     default statoil value is 0022, i.e. write access is removed from
     group and others.  

     The string is supposed to be in OCTAL representation (without any
     prefix characters).
  */
  if (config_item_set(config , UMASK_KEY)) {
    const char * string_mask = config_iget( config , UMASK_KEY , 0 , 0);
    mode_t umask_value;
    if (util_sscanf_octal_int( string_mask , &umask_value))
      site_config_set_umask( site_config , umask_value);
    else
      util_abort("%s: failed to parse:\"%s\" as a valid octal literal \n",__func__ , string_mask);
  }

  /* LSF options */
  {
    if (config_item_set(config , LSF_QUEUE_KEY))
      site_config_set_lsf_queue( site_config , config_iget( config , LSF_QUEUE_KEY , 0 , 0));
    
    if (config_item_set(config , LSF_RESOURCES_KEY)) {
      char * lsf_resource_request = config_alloc_joined_string(config , LSF_RESOURCES_KEY , " ");
      site_config_set_lsf_request( site_config , lsf_resource_request );
      free( lsf_resource_request );
    }
    
    if (config_item_set(config , MAX_RUNNING_LSF_KEY))
      site_config_set_max_running_lsf( site_config , config_iget_as_int( config , MAX_RUNNING_LSF_KEY , 0 , 0));
  }


  /* RSH options */
  {
    if (config_item_set( config , RSH_COMMAND_KEY ))
      site_config_set_rsh_command( site_config , config_iget(config , RSH_COMMAND_KEY , 0,0));
    
    if (config_item_set( config , MAX_RUNNING_RSH_KEY))
      site_config_set_max_running_rsh( site_config , config_iget_as_int( config , MAX_RUNNING_RSH_KEY , 0,0));

    /* Parsing the "host1:4" strings. */
    if (config_item_set( config , RSH_HOST_LIST_KEY)) {
      stringlist_type * rsh_host_list = config_alloc_complete_stringlist(config , RSH_HOST_LIST_KEY);
      int i;
      for (i=0; i < stringlist_get_size( rsh_host_list ); i++) {
        int     host_max_running;
        char ** tmp;
        char  * host;
        int     tokens;
        
        util_split_string( stringlist_iget( rsh_host_list , i) , ":" , &tokens , &tmp);
          if (tokens > 1) {
            if (!util_sscanf_int( tmp[tokens - 1] , &host_max_running))
              util_abort("%s: failed to parse out integer from: %s \n",__func__ , stringlist_iget( rsh_host_list , i));
            host = util_alloc_joined_string((const char **) tmp , tokens - 1 , ":");
          } else
            host = util_alloc_string_copy( tmp[0] );
          
          site_config_add_rsh_host( site_config , host , host_max_running);
          util_free_stringlist( tmp , tokens );
          free( host );
      }
      stringlist_free( rsh_host_list );
    }
  }

  if (config_item_set( config , QUEUE_SYSTEM_KEY)) {
    job_driver_type driver_type;
    {
      const char * queue_system = config_iget(config , QUEUE_SYSTEM_KEY , 0,0);
      if (strcmp(queue_system , "LSF") == 0) {
        driver_type = LSF_DRIVER;
      } else if (strcmp(queue_system , "RSH") == 0) 
        driver_type = RSH_DRIVER;
      else if (strcmp(queue_system , "LOCAL") == 0) 
        driver_type = LOCAL_DRIVER;
      else {
        util_abort("%s: queue system :%s not recognized \n",__func__ , queue_system);
        driver_type = NULL_DRIVER;
      }
    }
    site_config_set_job_queue__( site_config , driver_type );
  }
  
  /* Parsing local options */
  if (config_item_set( config , MAX_RUNNING_LOCAL_KEY))
    site_config_set_max_running_local( site_config , config_iget_as_int( config , MAX_RUNNING_LOCAL_KEY , 0,0));
  
  if (config_item_set(config , JOB_SCRIPT_KEY))
    site_config_set_job_script( site_config , config_iget( config , JOB_SCRIPT_KEY , 0 , 0));
  
  /* 
     When LSF is used several enviroment variables must be set - i.e.
     the calls to SETENV must come first. 
  */
  if (config_item_set(config , LICENSE_PATH_KEY))
    site_config_set_license_root_path( site_config , config_iget( config , LICENSE_PATH_KEY , 0 , 0));
  
  
  if (user_config) {
    site_config_install_job_queue(site_config );
    /* 
       The site_config_set_max_running_xxx() calls are collected here
       AFTER the call to site_config_set_job_queue__() to ensure that
       the correct (i.e. corresponding to the selected driver) size
       for the total queue is set in addition to the drivers specific
       max_running value.
    */
    site_config_set_max_running_lsf( site_config   , site_config->max_running_lsf );
    site_config_set_max_running_rsh( site_config   , site_config->max_running_rsh );
    site_config_set_max_running_local( site_config , site_config->max_running_local );
  }
}



void site_config_free(site_config_type * site_config) {
  ext_joblist_free( site_config->joblist );
  job_queue_free( site_config->job_queue );
  vector_free( site_config->user_action );
  
  stringlist_free( site_config->path_variables );
  stringlist_free( site_config->path_values );

  hash_free( site_config->rsh_host_list );
  hash_free( site_config->initial_variables );
  hash_free( site_config->env_variables );
  hash_free( site_config->initial_path_variables );

  if (site_config->__license_root_path != NULL)
    util_clear_directory( site_config->__license_root_path , true , true );
  
  util_safe_free( site_config->license_root_path );
  util_safe_free( site_config->__license_root_path );

  util_safe_free( site_config->job_script );
  util_safe_free( site_config->rsh_command );
  util_safe_free( site_config->lsf_queue_name );
  util_safe_free( site_config->lsf_request );
  free(site_config);
}


ext_joblist_type * site_config_get_installed_jobs( const site_config_type * site_config) {
  return site_config->joblist;
}


job_queue_type * site_config_get_job_queue( const site_config_type * site_config) {
  return site_config->job_queue;
}


void site_config_set_ens_size( site_config_type * site_config , int ens_size ) {
  job_queue_set_size( site_config->job_queue , ens_size );
}


/*****************************************************************/

void site_config_fprintf_config( const site_config_type * site_config , FILE * stream ) {
  fprintf( stream , CONFIG_COMMENTLINE_FORMAT );
  fprintf( stream , CONFIG_COMMENT_FORMAT , "Here comes system related information - which typically");
  fprintf( stream , CONFIG_COMMENT_FORMAT , "overrides information from the site-wide configuration file.");
  /* Starting with the user defined jobs. */
  {
    stringlist_type * joblist = ext_joblist_alloc_list( site_config->joblist );
    char * fmt_key     = util_alloc_sprintf(CONFIG_KEY_FORMAT , INSTALL_JOB_KEY);
    char * install_fmt = util_alloc_sprintf("%s%s%s" , fmt_key , CONFIG_VALUE_FORMAT , CONFIG_ENDVALUE_FORMAT);
    
    for (int i=0; i < stringlist_get_size( joblist ); i++) {
      ext_job_type * ext_job = ext_joblist_get_job( site_config->joblist ,stringlist_iget( joblist , i ));
      if (ext_job_is_private( ext_job ))
        ext_job_fprintf_config( ext_job , install_fmt , stream );
      
    }
    
    free( install_fmt );
    free( fmt_key );
  }

  
  /* Storing all the actions stored in user_action. */
  {
    for (int i=0; i< vector_get_size( site_config->user_action ); i++) {
      const stringlist_type * action = vector_iget_const( site_config->user_action , i );
      fprintf( stream , CONFIG_KEY_FORMAT , stringlist_iget( action , 0 ));
      for (int iarg = 1; iarg < stringlist_get_size( action ); iarg++)
        fprintf( stream , CONFIG_VALUE_FORMAT , stringlist_iget( action , iarg ));
      fprintf( stream , "\n");
    }
  }

  fprintf(stream , "\n\n");
}

/*****************************************************************/



void site_config_add_config_items( config_type * config , bool site_only) {
  config_item_type * item;

  item = config_add_item(config , QUEUE_SYSTEM_KEY , site_only , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);
  {
    stringlist_type * lsf_dep    = stringlist_alloc_argv_ref( (const char *[2]) {"LSF_QUEUE" , "MAX_RUNNING_LSF"}   , 2);
    stringlist_type * rsh_dep    = stringlist_alloc_argv_ref( (const char *[3]) {"RSH_HOST_LIST" , "RSH_COMMAND" , "MAX_RUNNING_RSH"} , 2);
    stringlist_type * local_dep  = stringlist_alloc_argv_ref( (const char *[1]) {"MAX_RUNNING_LOCAL"}   , 1);

    if (site_only) {
      config_item_set_common_selection_set( item , 3 , (const char *[3]) {"LSF" , "LOCAL" , "RSH"});
      config_item_set_required_children_on_value( item , "LSF"   , lsf_dep);
      config_item_set_required_children_on_value( item , "RSH"   , rsh_dep);
      config_item_set_required_children_on_value( item , "LOCAL" , local_dep);
    }

    stringlist_free(lsf_dep);
    stringlist_free(rsh_dep);
    stringlist_free(local_dep);
  }

  item = config_add_item(config , MAX_SUBMIT_KEY , false , false);   
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});


  /*
     You can set environment variables which will be applied to the
     run-time environment. Can unfortunately not use constructions
     like PATH=$PATH:/some/new/path, use the UPDATE_PATH function instead.
  */
  item = config_add_item(config , SETENV_KEY , false , true);
  config_item_set_argc_minmax(item , 2 , 2 , NULL);
  
  item = config_add_item(config , UMASK_KEY , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);

  /**
     UPDATE_PATH   LD_LIBRARY_PATH   /path/to/some/funky/lib

     Will prepend "/path/to/some/funky/lib" at the front of LD_LIBRARY_PATH.
  */
  item = config_add_item(config , UPDATE_PATH_KEY , false , true);
  config_item_set_argc_minmax(item , 2 , 2 , NULL);

  item = config_add_item( config , LICENSE_PATH_KEY , site_only , false );
  config_item_set_argc_minmax(item , 1 , 1, NULL );


  /*****************************************************************/
  /* Items related to running jobs with lsf/rsh/local ...          */

  /* These must be set IFF QUEUE_SYSTEM == LSF */
  item = config_add_item(config , LSF_QUEUE_KEY     , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);

  item = config_add_item(config , LSF_RESOURCES_KEY  , false , false);
  config_item_set_argc_minmax(item , 1 , -1 , NULL);

  item = config_add_item(config , MAX_RUNNING_LSF_KEY , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});


  /* These must be set IFF QUEUE_SYSTEM == RSH */
  config_add_item(config , RSH_HOST_LIST_KEY , false , false);
  item = config_add_item(config , RSH_COMMAND_KEY , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXECUTABLE});
  item = config_add_item(config , MAX_RUNNING_RSH_KEY , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});


  /* These must be set IFF QUEUE_SYSTEM == LOCAL */
  item = config_add_item(config , MAX_RUNNING_LOCAL_KEY , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});


  item = config_add_item(config , JOB_SCRIPT_KEY , site_only , false);
  config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});

  item = config_add_item(config , INSTALL_JOB_KEY , false , true);
  config_item_set_argc_minmax(item , 2 , 2 , (const config_item_types [2]) {CONFIG_STRING , CONFIG_EXISTING_FILE});
}
