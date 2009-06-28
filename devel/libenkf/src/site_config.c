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

/**
   This struct contains information which is specific to the site
   where this enkf instance is running. Pointers to the fields in this
   structure are passed on to e.g. the enkf_state->shared_info object,
   but this struct is the *OWNER* of this information, and hence
   responsible for booting and deleting these objects. 
*/

struct site_config_struct {
  ext_joblist_type 	* joblist;       /* The list of external jobs which have been installed. 
                   	                    These jobs will be the parts of the forward model. */
  job_queue_type   	* job_queue;     /* The queue instance which will run the external jobs. */
  char             	* image_viewer;  /* String pointing to an executable which can show images. */
  char                  * image_type;
  bool                    statoil_mode;  /* Quite obtrusive hack to support statoil_mode in the lsf_request. */
};


bool site_config_get_statoil_mode(const site_config_type * site_config ) {
  return site_config->statoil_mode;
}


static site_config_type * site_config_alloc_empty() {
  site_config_type * site_config = util_malloc( sizeof * site_config , __func__);

  site_config->joblist      = NULL;
  site_config->job_queue    = NULL;
  site_config->image_viewer = NULL;
  site_config->image_type   = NULL;
  
  return site_config;
}


static void site_config_set_image_viewer(site_config_type * site_config , const char * image_viewer) {
  site_config->image_viewer = util_realloc_string_copy(site_config->image_viewer , image_viewer );
}


const char * site_config_get_image_viewer(site_config_type * site_config) {
  return site_config->image_viewer;
}

static void site_config_set_image_type(site_config_type * site_config , const char * image_type) {
  site_config->image_type = util_realloc_string_copy(site_config->image_type , image_type );
}


const char * site_config_get_image_type(site_config_type * site_config) {
  return site_config->image_type;
}


static void site_config_install_job(site_config_type * site_config , const char * job_name , const char * install_file) {
  ext_joblist_add_job(site_config->joblist , job_name , install_file);
}


static void site_config_install_joblist(site_config_type * site_config , const config_type * config) {
  int  i;

  site_config->joblist = ext_joblist_alloc();
  stringlist_type *item_list = config_alloc_complete_stringlist(config , "INSTALL_JOB");

  for (i=0; i < stringlist_get_size(item_list); i+=2) 
    site_config_install_job(site_config , stringlist_iget(item_list , i) , stringlist_iget(item_list , i + 1));
  
  stringlist_free(item_list);
}



/**
   These functions can be called repeatedly if you should want to
   change driver characteristics run-time.
*/
static void site_config_install_LOCAL_job_queue(site_config_type * site_config , int ens_size , const char * job_script , int max_submit , int max_running) {
  basic_queue_driver_type * driver = local_driver_alloc();
  if (site_config->job_queue != NULL)
    job_queue_free( site_config->job_queue );
  
  site_config->job_queue = job_queue_alloc(ens_size , max_running , max_submit , job_script , driver);
}



static void site_config_install_RSH_job_queue(site_config_type * site_config , int ens_size , const char * job_script , int max_submit , int max_running , const char * rsh_command , const stringlist_type * rsh_host_list) {
  basic_queue_driver_type * driver = rsh_driver_alloc(rsh_command , rsh_host_list);
  if (site_config->job_queue != NULL)
    job_queue_free( site_config->job_queue );
  
  site_config->job_queue = job_queue_alloc(ens_size , max_running , max_submit , job_script , driver);
}


static void site_config_install_LSF_job_queue(site_config_type * site_config ,  int ens_size , const char * job_script , int max_submit , int max_running , const char * lsf_queue_name, const char * manual_lsf_request) {
  basic_queue_driver_type * driver = lsf_driver_alloc( lsf_queue_name );
  if (site_config->job_queue != NULL)
    job_queue_free( site_config->job_queue );
  
  site_config->job_queue   = job_queue_alloc(ens_size , max_running , max_submit , job_script , driver);
}





static void site_config_install_job_queue(site_config_type  * site_config , const config_type * config , int ens_size, bool * use_lsf) {
  const char * queue_system = config_get(config , "QUEUE_SYSTEM");
  const char * job_script   = config_get(config , "JOB_SCRIPT");
  int   max_submit          = strtol(config_get(config , "MAX_SUBMIT") , NULL , 10);
  *use_lsf                  = false;

  
  if (strcmp(queue_system , "LSF") == 0) {
    const char * lsf_queue_name = config_get(config , "LSF_QUEUE");
    char * lsf_resource_request = NULL;
    int max_running;
    
    
    if (config_has_set_item(config , "LSF_RESOURCES"))
      lsf_resource_request = config_alloc_joined_string(config , "LSF_RESOURCES" , " ");
    if (!util_sscanf_int(config_get(config , "MAX_RUNNING_LSF") , &max_running))
      util_abort("%s: internal error - \n",__func__);
    
    site_config_install_LSF_job_queue(site_config , ens_size , job_script , max_submit , max_running , lsf_queue_name , lsf_resource_request);
    util_safe_free(lsf_resource_request);
    *use_lsf = true;
  } else if (strcmp(queue_system , "RSH") == 0) {
    const char * rsh_command        = config_get(config , "RSH_COMMAND");
    stringlist_type * rsh_host_list = config_alloc_complete_stringlist(config , "RSH_HOST_LIST");
    int max_running;
    
    if (!util_sscanf_int(config_get(config , "MAX_RUNNING_RSH") , &max_running))
      util_abort("%s: internal error - \n",__func__);
    
    site_config_install_RSH_job_queue(site_config , ens_size , job_script , max_submit , max_running , rsh_command , rsh_host_list);
    stringlist_free( rsh_host_list );
  } else if (strcmp(queue_system , "LOCAL") == 0) {
    int max_running;
    
    if (!util_sscanf_int(config_get(config , "MAX_RUNNING_LOCAL") , &max_running))
      util_abort("%s: internal error - \n",__func__);
    site_config_install_LOCAL_job_queue(site_config , ens_size , job_script , max_submit , max_running);
  }
}


site_config_type * site_config_alloc(const config_type * config , int ens_size , bool * use_lsf) {
  const char * host_type    = config_get(config , "HOST_TYPE");
  site_config_type * site_config = site_config_alloc_empty();
  site_config_install_joblist(site_config , config);
  {
    int i;
    for (i = 0; i < config_get_occurences( config , "SETENV"); i++) {
      const stringlist_type * tokens = config_iget_stringlist_ref(config , "SETENV" , i);
      const char * var               = stringlist_iget( tokens , 0);
      const char * value             = stringlist_iget( tokens , 1);

      setenv( var , value , 1);
    }
    
    for (i=0; i < config_get_occurences( config, "UPDATE_PATH"); i++) {
      const stringlist_type * tokens = config_iget_stringlist_ref(config , "UPDATE_PATH" , i);
      const char * path              = stringlist_iget( tokens , 0);
      const char * value             = stringlist_iget( tokens , 1);
      
      util_update_path_var( path , value , false);
    }
  }
  /* 
     When LSF is used several enviroment variables must be set - i.e.
     the calls to SETENV must come first. 
  */
  if (strcmp(host_type , "STATOIL") == 0) 
    site_config->statoil_mode = true;
  else
    site_config->statoil_mode = false;
  
  site_config_install_job_queue(site_config , config , ens_size , use_lsf);
  site_config_set_image_viewer(site_config , config_get(config , "IMAGE_VIEWER"));
  site_config_set_image_type(site_config , config_get(config , "IMAGE_TYPE"));
  
  return site_config;
}

void site_config_free(site_config_type * site_config) {
  ext_joblist_free( site_config->joblist );
  job_queue_free( site_config->job_queue );
  free(site_config->image_viewer);
  free(site_config->image_type);
  free(site_config);
}

ext_joblist_type * site_config_get_installed_jobs( const site_config_type * site_config) {
  return site_config->joblist;
}

job_queue_type * site_config_get_job_queue( const site_config_type * site_config) {
  return site_config->job_queue;
}

