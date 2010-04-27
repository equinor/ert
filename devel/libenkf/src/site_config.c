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
#include <sys/stat.h>

/**
   This struct contains information which is specific to the site
   where this enkf instance is running. Pointers to the fields in this
   structure are passed on to e.g. the enkf_state->shared_info object,
   but this struct is the *OWNER* of this information, and hence
   responsible for booting and deleting these objects. 
*/

struct site_config_struct {
  ext_joblist_type 	* joblist;                /* The list of external jobs which have been installed. 
                   	                             These jobs will be the parts of the forward model. */
  job_queue_type   	* job_queue;              /* The queue instance which will run the external jobs. */
  hash_type             * env_variables;
  hash_type             * initial_variables;      /* Need to store the initial values so we can roll back. */
  hash_type             * initial_path_variables;
  stringlist_type       * path_variables;         /* We can update the same path variable several times - i.e. it can not be a hash table. */
  stringlist_type       * path_values;
  char                  * license_root_path;      /* The license_root_path value set by the user. */
  char                  * __license_root_path;    /* The license_root_path value actually used - includes a user/pid subdirectory. */
  mode_t                  umask;
  /*---------------------------------------------------------------*/
  int                     max_running_lsf;        /* Need to hold the detailed information about the         */
  char                  * lsf_queue_name;         /* various drivers here to be able to "hot-switch" driver. */
  char                  * lsf_request;  
  
  int                     max_running_rsh;
  hash_type             * rsh_host_list;
  char                  * rsh_command;

  int                     max_running_local;
  /*---------------------------------------------------------------*/
  bool                    statoil_mode;  /* Quite obtrusive hack to support statoil_mode in the lsf_request. */
};


bool site_config_get_statoil_mode(const site_config_type * site_config ) {
  return site_config->statoil_mode;
}


void site_config_set_umask( site_config_type * site_config , mode_t new_mask) {
  site_config->umask = new_mask;
  umask( new_mask );
}

mode_t site_config_get_umask( const site_config_type * site_config ) {
  return site_config->umask;
}


static site_config_type * site_config_alloc_empty() {
  site_config_type * site_config = util_malloc( sizeof * site_config , __func__);
  
  site_config->joblist                = NULL;
  site_config->job_queue              = NULL;

  site_config->lsf_queue_name         = NULL;
  site_config->lsf_request            = NULL;
  site_config->rsh_host_list          = hash_alloc();
  site_config->rsh_command            = NULL;
  site_config->license_root_path      = NULL;
  site_config->__license_root_path    = NULL;

  site_config->max_running_local      = 0;
  site_config->max_running_lsf        = 0;
  site_config->max_running_rsh        = 0;

  /* Some hooops to get the current umask. */ 
  site_config->umask                  = umask( 0 );
  site_config_set_umask( site_config , site_config->umask );

  site_config->initial_variables      = hash_alloc();
  site_config->env_variables          = hash_alloc();
  site_config->path_variables         = stringlist_alloc_new();
  site_config->path_values            = stringlist_alloc_new();
  site_config->initial_path_variables = hash_alloc();
  
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
  site_config->__license_root_path = util_realloc_sprintf(site_config->__license_root_path , "%s%c%s%c%d" , UTIL_PATH_SEP_CHAR , getenv("USER") , UTIL_PATH_SEP_CHAR , getpid());
}



/**
   Will return 0 if the job is added correctly, and a non-zero (not
   documented ...) error code if the job is not added. 
*/

int site_config_install_job(site_config_type * site_config , const char * job_name , const char * install_file) {
  ext_job_type * new_job = ext_job_fscanf_alloc(job_name , site_config->__license_root_path , false, install_file) ;
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



static void site_config_install_joblist(site_config_type * site_config , const config_type * config) {
  int  i;

  site_config->joblist = ext_joblist_alloc( config_iget( config , "LICENSE_PATH" , 0 , 0));
  stringlist_type *item_list = config_alloc_complete_stringlist(config , "INSTALL_JOB");

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
  basic_queue_driver_type * driver = lsf_driver_alloc( site_config->lsf_queue_name );
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
  if (job_queue_get_driver_type( site_config->job_queue ) == LSF_DRIVER)
    job_queue_set_max_running( site_config->job_queue , max_running_lsf );
}

int site_config_get_max_running_lsf( const site_config_type * site_config ) {
  return site_config->max_running_lsf;
}

void site_config_set_max_running_rsh( site_config_type * site_config , int max_running_rsh) {
  site_config->max_running_rsh = max_running_rsh;
  if (job_queue_get_driver_type( site_config->job_queue ) == RSH_DRIVER)
    job_queue_set_max_running( site_config->job_queue , max_running_rsh );
}

int site_config_get_max_running_rsh( const site_config_type * site_config) {
  return site_config->max_running_rsh;
}

void site_config_set_max_running_local( site_config_type * site_config , int max_running_local) {
  site_config->max_running_local = max_running_local;
  if (job_queue_get_driver_type( site_config->job_queue ) == LOCAL_DRIVER)
    job_queue_set_max_running( site_config->job_queue , max_running_local );
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


void site_config_add_rsh_host( const site_config_type * site_config , const char * rsh_host , int max_running ) {
  hash_insert_int( site_config->rsh_host_list , rsh_host , max_running );
}


/*****************************************************************/

void site_config_set_lsf_queue( site_config_type * site_config , const char * lsf_queue) {
  site_config->lsf_queue_name = util_realloc_string_copy( site_config->lsf_queue_name , lsf_queue);
}

const char * site_config_get_lsf_queue( const site_config_type * site_config ) {
  return site_config->lsf_queue_name;
}

void site_config_set_lsf_request( site_config_type * site_config , const char * lsf_request) {
  site_config->lsf_request = util_realloc_string_copy( site_config->lsf_request , lsf_request);
}

const char * site_config_get_lsf_request( const site_config_type * site_config ) {
  return site_config->lsf_request;
}

/*****************************************************************/

void site_config_set_rsh_command( site_config_type * site_config , const char * rsh_command) {
  site_config->rsh_command = util_realloc_string_copy( site_config->rsh_command  , rsh_command);
}

const char * site_config_get_rsh_command( const site_config_type * site_config ) {
  return site_config->rsh_command;
}





/*****************************************************************/


const char * site_config_get_queue_name( const site_config_type * site_config ) {
  return job_queue_get_driver_name( site_config->job_queue );
}


static void site_config_set_job_queue__( site_config_type * site_config , job_driver_type driver_type) {
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


void site_config_set_job_queue( site_config_type * site_config , const char * queue_name ) {
  job_driver_type driver_type = job_queue_lookup_driver_name( queue_name );
  site_config_set_job_queue__( site_config , driver_type );
}

bool site_config_queue_is_running( const site_config_type * site_config ) {
  return job_queue_is_running( site_config->job_queue );
}


void site_config_set_job_script( site_config_type * site_config , const char * job_script ) {
  job_queue_set_run_cmd( site_config->job_queue  , job_script );
}


const char * site_config_get_job_script( const site_config_type * site_config ) {
  return job_queue_get_run_cmd( site_config->job_queue );
}


void site_config_set_max_submit( site_config_type * site_config , int max_submit ) {
  job_queue_set_max_submit(site_config->job_queue , max_submit);
}

int site_config_get_max_submit(const site_config_type * site_config ) {
  return job_queue_get_max_submit(site_config->job_queue);
}


static void site_config_install_job_queue(site_config_type  * site_config , const config_type * config , bool * use_lsf) {
  const char * job_script   = config_iget(config , "JOB_SCRIPT" , 0,0);
  int   max_submit          = config_iget_as_int(config , "MAX_SUBMIT" , 0,0);
  int   max_running_local;
  int   max_running_lsf;
  int   max_running_rsh;
  job_driver_type driver_type;

  *use_lsf               = false;
  site_config->job_queue = job_queue_alloc(0 , 0 , max_submit , job_script);

  /* 
     All the various driver options are set, unconditionally of which
     driver is actually selected in the end.
  */

  /* LSF options. */
  {
    const char * lsf_queue_name = config_iget(config , "LSF_QUEUE" , 0,0);
    char * lsf_resource_request = NULL;

    max_running_lsf = config_iget_as_int( config , "MAX_RUNNING_LSF" , 0,0);
    if (config_has_set_item(config , "LSF_RESOURCES"))
      lsf_resource_request = config_alloc_joined_string(config , "LSF_RESOURCES" , " ");
    
    site_config_set_lsf_queue( site_config , lsf_queue_name );
    site_config_set_lsf_request( site_config , lsf_resource_request );  /* Not really in use at the moment ... */
    util_safe_free(lsf_resource_request);
  }
    

  /* RSH options */
  {
    const char * rsh_command        = config_iget(config , "RSH_COMMAND" , 0,0);
    
    site_config_set_rsh_command( site_config , rsh_command );
    max_running_rsh = config_iget_as_int( config , "MAX_RUNNING_RSH" , 0,0);

    /* Parsing the "host1:4" strings. */
    {
      
      stringlist_type * rsh_host_list = config_alloc_complete_stringlist(config , "RSH_HOST_LIST");
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

  /* Parsing local options */
  max_running_local = config_iget_as_int( config , "MAX_RUNNING_LOCAL" , 0,0);
  
  
  {
    const char * queue_system = config_iget(config , "QUEUE_SYSTEM" , 0,0);
    if (strcmp(queue_system , "LSF") == 0) {
      *use_lsf = true;
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

  /* 
     The site_config_set_max_running_xxx() calls are collected here
     AFTER the call to site_config_set_job_queue__() to ensure that
     the correct (i.e. corresponding to the selected driver) size for
     the queue is set in addition to the drivers specific max_running
     value.
  */
  site_config_set_max_running_lsf( site_config , max_running_lsf );
  site_config_set_max_running_rsh( site_config , max_running_rsh );
  site_config_set_max_running_local( site_config , max_running_local );
}




site_config_type * site_config_alloc(const config_type * config , bool * use_lsf) {
  const char * host_type         = config_iget(config , "HOST_TYPE" , 0,0);
  site_config_type * site_config = site_config_alloc_empty();
  site_config_install_joblist(site_config , config);
  {
    int i;
    for (i = 0; i < config_get_occurences( config , "SETENV"); i++) {
      const stringlist_type * tokens = config_iget_stringlist_ref(config , "SETENV" , i);
      const char * var               = stringlist_iget( tokens , 0);
      const char * value             = stringlist_iget( tokens , 1);

      site_config_setenv( site_config , var , value );
    }
    
    for (i=0; i < config_get_occurences( config, "UPDATE_PATH"); i++) {
      const stringlist_type * tokens = config_iget_stringlist_ref(config , "UPDATE_PATH" , i);
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
  if (config_item_set(config , "UMASK")) {
    const char * string_mask = config_iget( config , "UMASK" , 0 , 0);
    mode_t umask_value;
    if (util_sscanf_octal_int( string_mask , &umask_value))
      site_config_set_umask( site_config , umask_value);
    else
      util_abort("%s: failed to parse:\"%s\" as a valid octal literal \n",__func__ , string_mask);
  }

  /* 
     When LSF is used several enviroment variables must be set - i.e.
     the calls to SETENV must come first. 
  */
  if (strcmp(host_type , "STATOIL") == 0) 
    site_config->statoil_mode = true;
  else
    site_config->statoil_mode = false;
  
  site_config_install_job_queue(site_config , config , use_lsf);
  return site_config;
}



void site_config_free(site_config_type * site_config) {
  ext_joblist_free( site_config->joblist );
  job_queue_free( site_config->job_queue );
  
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
