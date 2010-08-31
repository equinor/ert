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
#include <lsf_driver.h>
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
   user's config file should of course be saved. This is 'solved' with
   many fields having a xxx_site duplicate, where the xxx_site is only
   updated during the initial parsing of the site-config file; when
   the flag user_mode is set to true the xxx_site fields are not
   updated. When saving only fields which are different from their
   xxx_site counterpart are stored.
*/


struct site_config_struct {
  ext_joblist_type 	* joblist;                /* The list of external jobs which have been installed. 
                   	                             These jobs will be the parts of the forward model. */
  hash_type             * env_variables_user;     /* The environment variables set in the user config file. */
  hash_type             * env_variables_site;     /* The environment variables set in site_config file - not exported. */ 
  
  mode_t                  umask;
  mode_t                  umask_site;
  
  
  char                  * license_root_path;      /* The license_root_path value set by the user. */
  char                  * license_root_path_site; /* The license_root_path value set by the site. */
  char                  * __license_root_path;    /* The license_root_path value actually used - includes a user/pid subdirectory. */

  hash_type             * path_variables_site;    /* We store this so we can roll back when all user settings are cleared. */
  stringlist_type       * path_variables_user;    /* We can update the same path variable several times - i.e. it can not be a hash table. */
  stringlist_type       * path_values_user;
  
  
  int                     max_running_lsf;        
  int                     max_running_lsf_site;   /* Need to hold the detailed information about the         */
  char                  * lsf_queue_name;         /* various drivers here to be able to "hot-switch" driver. */
  char                  * lsf_request;  
  char                  * lsf_queue_name_site;         
  char                  * lsf_request_site;  
  char                  * remote_lsf_server;      /* Not in use */
  
                           
  hash_type             * rsh_host_list;          /* rsh_host_list is NOT updated when parsing the site_config file. */
  int                     max_running_rsh_site;
  int                     max_running_rsh;
  char                  * rsh_command;
  char                  * rsh_command_site;

  int                     max_running_local;
  int                     max_running_local_site;

  job_driver_type         driver_type;
  job_driver_type         driver_type_site;
  int                     max_submit;             
  int                     max_submit_site;             
  char                  * job_script;            
  char                  * job_script_site;            

  int                     num_cpu;                /* The number of cpu's used to run the forward model - currently only relevant for ECLIPSE and LSF; read automatically from the ECLIPSE data file. */
  job_queue_type   	* job_queue;              /* The queue instance which will run the external jobs. */
  bool                    user_mode;
};


void site_config_set_num_cpu( site_config_type * site_config , int num_cpu ) {
  job_queue_set_num_cpu( site_config->job_queue , num_cpu );
}


void site_config_set_umask( site_config_type * site_config , mode_t new_mask) {
  umask( new_mask );

  site_config->umask = new_mask;
  if (!site_config->user_mode) 
    site_config->umask_site = new_mask;
  
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
  
  site_config->remote_lsf_server      = NULL;
  site_config->lsf_queue_name         = NULL;
  site_config->lsf_queue_name_site    = NULL;
  site_config->lsf_request            = NULL;
  site_config->lsf_request_site       = NULL;
  site_config->rsh_host_list          = hash_alloc();
  site_config->rsh_command            = NULL;
  site_config->rsh_command_site       = NULL;
  site_config->license_root_path      = NULL;
  site_config->license_root_path_site = NULL;
  site_config->__license_root_path    = NULL;
  site_config->job_script             = NULL;  
  site_config->job_script_site        = NULL;  

  site_config->user_mode             = false;
  site_config->num_cpu                = 0;
  site_config->max_running_local      = 0;
  site_config->max_running_lsf        = 0;
  site_config->max_running_rsh        = 0;
  site_config->driver_type            = NULL_DRIVER;

  /* Some hooops to get the current umask. */ 
  site_config->umask                  = umask( 0 );
  site_config_set_umask( site_config , site_config->umask );

  site_config->env_variables_user     = hash_alloc();
  site_config->env_variables_site     = hash_alloc();
  
  site_config->path_variables_user    = stringlist_alloc_new();
  site_config->path_values_user       = stringlist_alloc_new();
  site_config->path_variables_site    = hash_alloc();
  
  site_config_set_max_submit( site_config , DEFAULT_MAX_SUBMIT );
  return site_config;
}


const char * site_config_get_license_root_path( const site_config_type * site_config ) {
  return site_config->license_root_path;
}


/**
   Observe that this variable can not "really" be set to different
   values during a simulation, when creating ext_job instances they
   will store a pointer to this variable on creation, if the variable
   is later changed they will be left with a dangling copy. That is
   not particularly elegant, however it should nonetheless work.
*/

void site_config_set_license_root_path( site_config_type * site_config , const char * license_root_path) {
  util_make_path( license_root_path );
  {
    char * full_license_root_path = util_alloc_realpath( license_root_path );
    {
      /**
         Appending /user/pid to the license root path. Everything
         including the pid is removed when exiting (gracefully ...).
         
         Dangling license directories after a crash can just be removed.
      */
      site_config->license_root_path   = util_realloc_string_copy( site_config->license_root_path , full_license_root_path );
      site_config->__license_root_path = util_realloc_sprintf(site_config->__license_root_path , "%s%c%s%c%d" , full_license_root_path , UTIL_PATH_SEP_CHAR , getenv("USER") , UTIL_PATH_SEP_CHAR , getpid());
      
      if (!site_config->user_mode)
        site_config->license_root_path_site = util_realloc_string_copy( site_config->license_root_path_site , full_license_root_path );
    }
    free( full_license_root_path );
  }
}



void site_config_init_user_mode( site_config_type * site_config ) {
  site_config->user_mode = true;
}


/**
   Will return 0 if the job is added correctly, and a non-zero (not
   documented ...) error code if the job is not added. 
*/

int site_config_install_job(site_config_type * site_config , const char * job_name , const char * install_file) {
  ext_job_type * new_job = ext_job_fscanf_alloc(job_name , site_config->__license_root_path , site_config->user_mode , install_file) ;
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
  return site_config->env_variables_user;
}


/**
   Will only return the user-set variables. The variables set in the
   site config are hidden.
*/

stringlist_type * site_config_get_path_variables( const site_config_type * site_config ) {
  return site_config->path_variables_user;
}

stringlist_type * site_config_get_path_values( const site_config_type * site_config ) {
  return site_config->path_values_user;
}

/**
   Observe that the value inserted in the internal hash tables is the
   interpolated value returned from util_setenv(), where $VAR
   expressions have been expanded.
*/
   
void site_config_setenv( site_config_type * site_config , const char * variable, const char * __value) {
  const char * value = util_setenv( variable , __value );

  if (site_config->user_mode) {
    /* In the table meant for user-export we store the literal $var strings. */
    hash_insert_hash_owned_ref( site_config->env_variables_user , variable , util_alloc_string_copy( __value ) , free);
    
    if (!hash_has_key( site_config->env_variables_site , variable))
      hash_insert_ref( site_config->env_variables_site , variable , NULL);   /* We insert a NULL so we can recover a unsetenv() in _clear_env(). */
  } else
    hash_insert_hash_owned_ref( site_config->env_variables_site , variable , util_alloc_string_copy( value ) , free);
}



/**
   Clears all the environment variables set by the user. This is done
   is follows:
     
     1. Iterate through the table config->env_variables_user and call
        unsetenv() on all of them
   
     2. Iterate through the table config->env_variables_site and call
        setenv() on all of them.
        
   This way the environment should be identical to what it is after
   the site parsing is completed.
*/


void site_config_clear_env( site_config_type * site_config ) {
  /* 1: Clearing the user_set variables. */
  {
    hash_iter_type * hash_iter = hash_iter_alloc( site_config->env_variables_user );
    while (!hash_iter_is_complete( hash_iter )) {
      const char * var       = hash_iter_get_next_key( hash_iter );
      unsetenv( var );
    }
    hash_iter_free( hash_iter );
    hash_clear( site_config->env_variables_user );
  }
  

  /* 2: Recovering the site_set variables. */
  {
    hash_iter_type * hash_iter = hash_iter_alloc( site_config->env_variables_site );
    while (!hash_iter_is_complete( hash_iter )) {
      const char * var       = hash_iter_get_next_key( hash_iter );
      const char * value     = hash_get( site_config->env_variables_site , var );
      util_setenv( var , value );    /* Will call unsetenv if value == NULL */
    }
    hash_iter_free( hash_iter );
  }  
}


void site_config_clear_pathvar( site_config_type * site_config ) {
  stringlist_clear( site_config->path_variables_user );
  stringlist_clear( site_config->path_values_user );
  {
    /* Recover the original values. */
    hash_iter_type * hash_iter = hash_iter_alloc( site_config->path_variables_site );
    while (!hash_iter_is_complete( hash_iter )) {
      const char * var       = hash_iter_get_next_key( hash_iter );
      const char * site_value = hash_get( site_config->path_variables_site , var );
      
      if (site_value == NULL)
        unsetenv( var );
      else
        setenv( var , site_value , 1 );
    }
  }
}


void site_config_update_pathvar( site_config_type * site_config , const char * pathvar , const char * value) {
  if (site_config->user_mode) {
    stringlist_append_copy( site_config->path_variables_user , pathvar );
    stringlist_append_copy( site_config->path_values_user    , value   );
    
    if (!hash_has_key( site_config->path_variables_site , pathvar )) 
      hash_insert_ref( site_config->path_variables_site , pathvar , NULL); /* This path variable has not been touched in the
                                                                              site_config. We store a NULL, so can roll back
                                                                              (i.e. call unsetenv()). */
  } 
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
  basic_queue_driver_type * driver = lsf_driver_alloc( site_config->lsf_queue_name , site_config->lsf_request , /* site_config->remote_lsf_server ,*/ site_config->num_cpu);
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

  if (!site_config->user_mode) 
    site_config->max_running_lsf_site = max_running_lsf;
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
  if (!site_config->user_mode) 
    site_config->max_running_rsh_site = max_running_rsh;
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
  if (!site_config->user_mode) 
    site_config->max_running_local_site = max_running_local;
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
}


/*****************************************************************/

void site_config_set_lsf_queue( site_config_type * site_config , const char * lsf_queue) {
  site_config->lsf_queue_name = util_realloc_string_copy( site_config->lsf_queue_name , lsf_queue);
  if (!site_config->user_mode) 
    site_config->lsf_queue_name_site = util_realloc_string_copy( site_config->lsf_queue_name_site , lsf_queue);

  if (site_config->job_queue != NULL) {
    job_driver_type current_driver = job_queue_get_driver_type( site_config->job_queue );
    if (current_driver == LSF_DRIVER) {  /* Must push the update down to the driver. */
      lsf_driver_type * lsf_driver = lsf_driver_safe_cast( job_queue_get_driver( site_config->job_queue ));
      lsf_driver_set_queue_name( lsf_driver , lsf_queue );
    }
  }
}
      


const char * site_config_get_lsf_queue( const site_config_type * site_config ) {
  return site_config->lsf_queue_name;
}



void site_config_set_lsf_request( site_config_type * site_config , const char * lsf_request) {
  site_config->lsf_request = util_realloc_string_copy( site_config->lsf_request , lsf_request);
  if (!site_config->user_mode) 
    site_config->lsf_request_site = util_realloc_string_copy( site_config->lsf_request_site , lsf_request);

  if (site_config->job_queue != NULL) {
    job_driver_type current_driver = job_queue_get_driver_type( site_config->job_queue );
    if (current_driver == LSF_DRIVER) {  /* Must push the update down to the driver. */
      lsf_driver_type * lsf_driver = lsf_driver_safe_cast( job_queue_get_driver( site_config->job_queue ));
      lsf_driver_set_resource_request( lsf_driver , lsf_request );
    }
  }
}

const char * site_config_get_lsf_request( const site_config_type * site_config ) {
  return site_config->lsf_request;
}

/*****************************************************************/

void site_config_set_rsh_command( site_config_type * site_config , const char * rsh_command) {
  site_config->rsh_command = util_realloc_string_copy( site_config->rsh_command  , rsh_command);
  if (!site_config->user_mode) 
    site_config->rsh_command_site = util_realloc_string_copy( site_config->rsh_command_site  , rsh_command);
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
  if (!site_config->user_mode)
    site_config->driver_type_site = driver_type;
}



void site_config_set_job_queue( site_config_type * site_config , const char * queue_name ) {
  job_driver_type driver_type = job_queue_lookup_driver_name( queue_name );
  site_config_set_job_queue__( site_config , driver_type );
}

bool site_config_queue_is_running( const site_config_type * site_config ) {
  return job_queue_is_running( site_config->job_queue );
}

/**
   The job_script might be a relative path, and the cwd changes during
   execution, i.e. it is essential to get hold of the full path.
*/

void site_config_set_job_script( site_config_type * site_config , const char * job_script ) {
  char * job_script_full_path = util_alloc_realpath( job_script );
  {
    site_config->job_script = util_realloc_string_copy( site_config->job_script , job_script_full_path );
    if (site_config->job_queue != NULL)
      job_queue_set_run_cmd( site_config->job_queue  , site_config->job_script );
    
    if (!site_config->user_mode) 
      site_config->job_script_site = util_realloc_string_copy( site_config->job_script_site , site_config->job_script );
  }
  free( job_script_full_path );
}


const char * site_config_get_job_script( const site_config_type * site_config ) {
  return job_queue_get_run_cmd( site_config->job_queue );
}


void site_config_set_max_submit( site_config_type * site_config , int max_submit ) {
  site_config->max_submit = max_submit;
  if (!site_config->user_mode)
    site_config->max_submit_site = max_submit;
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
    if (user_config) {
      if (config_item_set( config , RSH_HOST_KEY)) {
        stringlist_type * rsh_host_list = config_alloc_complete_stringlist(config , RSH_HOST_KEY);
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
  
  stringlist_free( site_config->path_variables_user );
  stringlist_free( site_config->path_values_user );
  hash_free( site_config->path_variables_site );

  hash_free( site_config->rsh_host_list );
  hash_free( site_config->env_variables_site );
  hash_free( site_config->env_variables_user );

  if (site_config->__license_root_path != NULL)
    util_clear_directory( site_config->__license_root_path , true , true );
  
  util_safe_free( site_config->license_root_path );
  util_safe_free( site_config->license_root_path_site );
  util_safe_free( site_config->__license_root_path );

  util_safe_free( site_config->remote_lsf_server );

  util_safe_free( site_config->job_script );
  util_safe_free( site_config->job_script_site );
  util_safe_free( site_config->rsh_command );
  util_safe_free( site_config->rsh_command_site );
  util_safe_free( site_config->lsf_queue_name );
  util_safe_free( site_config->lsf_queue_name_site );
  util_safe_free( site_config->lsf_request );
  util_safe_free( site_config->lsf_request_site );
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

  
  /* Storing the env variables set with SETENV */
  {
    hash_iter_type * iter = hash_iter_alloc( site_config->env_variables_user );
    while (!hash_iter_is_complete( iter )) {
      const char * var        = hash_iter_get_next_key( iter );
      const char * user_value = hash_get( site_config->env_variables_user , var );
      const char * site_value = hash_safe_get( site_config->env_variables_site , var );
      
      if (!util_string_equal( user_value , site_value)) {
        fprintf(stream , CONFIG_KEY_FORMAT      , SETENV_KEY );
        fprintf(stream , CONFIG_VALUE_FORMAT    , var );
        fprintf(stream , CONFIG_ENDVALUE_FORMAT , user_value );
      }
    }
  }

  /* Storing the driver type setting: */
  if ( site_config->driver_type != site_config->driver_type_site) {
    fprintf(stream , CONFIG_KEY_FORMAT , QUEUE_SYSTEM_KEY );
    fprintf(stream , CONFIG_ENDVALUE_FORMAT , site_config_get_queue_name( site_config ));
  }
  
  /* Storing UMASK setting */
  if ( site_config->umask != site_config->umask_site) {
    fprintf(stream , CONFIG_KEY_FORMAT , UMASK_KEY );
    fprintf(stream , "%o\n" , site_config->umask );
  }

  /* Storing MAX_SUBMIT setting */
  if ( site_config->max_submit != site_config->max_submit_site) {
    fprintf(stream , CONFIG_KEY_FORMAT , MAX_SUBMIT_KEY );
    fprintf(stream , "%d\n" , site_config->max_submit );
  }
  
  /* Storing LICENSE_ROOT_PATH */
  if (!util_string_equal( site_config->license_root_path , site_config->license_root_path_site)) {
    fprintf(stream , CONFIG_KEY_FORMAT      , LICENSE_PATH_KEY );
    fprintf(stream , CONFIG_ENDVALUE_FORMAT , site_config->license_root_path);
  }

  /* Storing jobscript */
  if (!util_string_equal( site_config->job_script , site_config->job_script_site)) {
    fprintf(stream , CONFIG_KEY_FORMAT      , LICENSE_PATH_KEY );
    fprintf(stream , CONFIG_ENDVALUE_FORMAT , site_config->job_script);
  }

  /* Storing local settings. */
  if (site_config->max_running_local != site_config->max_running_local_site) {
    fprintf(stream , CONFIG_KEY_FORMAT      , MAX_RUNNING_LOCAL_KEY );
    fprintf(stream , CONFIG_INT_FORMAT , site_config->max_running_local);
    fprintf( stream , "\n");
  }

  /* Storing LSF settings. */
  {
    if (site_config->max_running_lsf != site_config->max_running_lsf_site) {
      fprintf(stream , CONFIG_KEY_FORMAT , MAX_RUNNING_LSF_KEY );
      fprintf(stream , CONFIG_INT_FORMAT , site_config->max_running_lsf);
      fprintf( stream , "\n");
    } 

    if (!util_string_equal( site_config->lsf_queue_name , site_config->lsf_queue_name_site)) {
      fprintf(stream , CONFIG_KEY_FORMAT      , LSF_QUEUE_KEY );
      fprintf(stream , CONFIG_ENDVALUE_FORMAT , site_config->lsf_queue_name);
    }

    if (!util_string_equal( site_config->lsf_request , site_config->lsf_request_site)) {
      fprintf(stream , CONFIG_KEY_FORMAT      , LSF_RESOURCES_KEY );
      fprintf(stream , CONFIG_ENDVALUE_FORMAT , site_config->lsf_request);
    }
  }

  
  /* Storing RSH settings. */
  {
    if (site_config->max_running_rsh != site_config->max_running_rsh_site) {
      fprintf(stream , CONFIG_KEY_FORMAT      , MAX_RUNNING_RSH_KEY );
      fprintf(stream , CONFIG_INT_FORMAT , site_config->max_running_rsh);
      fprintf( stream , "\n");
    }
    
    if (!util_string_equal( site_config->rsh_command , site_config->rsh_command_site)) {
      fprintf(stream , CONFIG_KEY_FORMAT      , LICENSE_PATH_KEY );
      fprintf(stream , CONFIG_ENDVALUE_FORMAT , site_config->rsh_command);
    }

    {
      hash_iter_type * iter = hash_iter_alloc( site_config->rsh_host_list );
      while (!hash_iter_is_complete( iter )) {
        const char * host_name = hash_iter_get_next_key( iter );
        fprintf(stream , CONFIG_KEY_FORMAT      , RSH_HOST_KEY );
        fprintf(stream , "%s:%d\n"  , host_name , hash_get_int( site_config->rsh_host_list , host_name));
      }
      hash_iter_free( iter );
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
    stringlist_type * rsh_dep    = stringlist_alloc_argv_ref( (const char *[3]) {"RSH_HOST"  , "RSH_COMMAND" , "MAX_RUNNING_RSH"} , 2);
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
  config_item_set_envvar_expansion( item , false );   /* Do not expand $VAR expressions (that is done in util_setenv()). */
  
  item = config_add_item(config , UMASK_KEY , false , false);
  config_item_set_argc_minmax(item , 1 , 1 , NULL);

  /**
     UPDATE_PATH   LD_LIBRARY_PATH   /path/to/some/funky/lib

     Will prepend "/path/to/some/funky/lib" at the front of LD_LIBRARY_PATH.
  */
  item = config_add_item(config , UPDATE_PATH_KEY , false , true);
  config_item_set_argc_minmax(item , 2 , 2 , NULL);
  config_item_set_envvar_expansion( item , false );   /* Do not expand $VAR expressions (that is done in util_setenv()). */

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
  if (!site_only)
    config_add_item(config , RSH_HOST_KEY , false , false);  /* Only added when user parse. */
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
