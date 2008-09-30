#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <util.h>
#include <hash.h>
#include <multz_config.h>
#include <enkf_config_node.h>
#include <path_fmt.h>
#include <ecl_static_kw_config.h>
#include <enkf_types.h>
#include <well_config.h>
#include <field_config.h>
#include <equil_config.h>
#include <multz_config.h>
#include <multflt_config.h>
#include <well_obs.h>
#include <thread_pool.h>
#include <obs_node.h>
#include <obs_data.h>
#include <history.h>
#include <meas_matrix.h>
#include <enkf_state.h>  
#include <analysis.h>
#include <enkf_obs.h>
#include <sched_file.h>
#include <enkf_fs.h>
#include <void_arg.h>
#include <gen_kw_config.h>
#include <history.h>
#include <node_ctype.h>
#include <pthread.h>
#include <job_queue.h>
#include <msg.h>
#include <stringlist.h>
#include <enkf_main.h>
#include <enkf_serialize.h>
#include <config.h>
#include <local_driver.h>
#include <rsh_driver.h>
#include <lsf_driver.h>
#include <history.h>
#include <enkf_sched.h>
#include <set.h>

#include "enkf_defaults.h"


/**
   This structure contains files which are set during the
   initialisation process, but not in the rest of the simulation. For
   instance the user specifies a grid file in the CONFIG_FILE, but
   after the bootstrapping is complete the EnKF program does not use
   this filename. Instead the grid information is internalized in an
   ecl_grid_type structure, and the filename is no longer
   interesting. Similar with all the other fields in this structure.
*/


struct config_tmp_files_struct {
  char * schedule_src_file;
  char * grid_file;
  char * enkf_sched_file;
};




/**
   This struct contains information which is specific to the site
   where this enkf instance is running. Pointers to the fields in this
   structure are passed on to e.g. the enkf_state->shared_info object,
   but this struct is the *OWNER* of this information, and hence
   responsible for booting and deleting these objects. 
*/

typedef struct {
  ext_joblist_type * joblist;       /* The list of external jobs which have been installed. 
                                       These jobs will be the parts of the forward model. */
  job_queue_type   * job_queue;     /* The queue instance which will run the external jobs. */
  char             * image_viewer;  /* String pointing to an executable which can show images. */
} site_config_type;



/**
   Eclipse info
*/

typedef struct {
  bool                endian_swap;            /* Should an endian swap be performed - (in general: YES) */
  bool                fmt_file;
  bool                unified;

  path_fmt_type     * eclbase;
  ecl_grid_type     * grid;
  sched_file_type   * sched_file;
  bool                include_all_static_kw;  /* If true all static keywords are stored.*/ 
  set_type          * static_kw_set;          /* Minimum set of static keywords which must be included to make valid restart files. */
} eclipse_config_type;




/**
   This struct contains configuration which is specific to this
   particular model/run. Much of the information is actually accessed
   directly through the enkf_state object; but this struct is the
   owner of the information, and responsible for allocating/freeing
   it.
*/

typedef struct {
  int                 ens_size; 
  enkf_fs_type      * ensemble_dbase;

  
  history_type      * history;
  time_t              start_time;

  
  path_fmt_type     * result_path;
  path_fmt_type     * runpath;
  enkf_sched_type   * enkf_sched;
} model_config_type;



/**
   This object should contain **everything** needed to run a enkf
   simulation. A way to wrap up all available information/state and
   pass it around.
*/

#define ENKF_MAIN_ID 8301

struct enkf_main_struct {
  int                 __id;       /* Used for type-checking run-time casts. */
  enkf_config_type   *config;
  job_queue_type     *job_queue;
  ecl_grid_type      *grid;
  enkf_obs_type      *obs;
  meas_matrix_type   *meas_matrix;
  obs_data_type      *obs_data;      /* Should ideally contain the hist object. */
  enkf_state_type   **ensemble;
  sched_file_type    *sched_file;
  history_type       *hist;
  enkf_fs_type       *fs;
  run_mode_type       run_mode;      /* Is this an enkf assimilation, or an ensemble experiment, or .. */

  lock_mode_type      runlock_mode;  /* How/if should the runpath directories be locked */
  char               *lock_path;     /* Path containg lock files for the forward run. */ 
}; 


/*****************************************************************/


static site_config_type * site_config_alloc_empty() {
  site_config_type * site_config = util_malloc( sizeof * site_config , __func__);

  site_config->joblist      = NULL;
  site_config->job_queue    = NULL;
  site_config->image_viewer = NULL;

  return site_config;
}


static void site_config_set_image_viewer(site_config_type * site_config , const char * image_viewer) {
  site_config->image_viewer = util_realloc_string_copy(site_config->image_viewer , image_viewer );
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


static void site_config_free(site_config_type * site_config) {
  ext_joblist_free( site_config->joblist );
  job_queue_free( site_config->job_queue );
  free(site_config->image_viewer);
  free(site_config);
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



static void site_config_install_LSF_job_queue(site_config_type * site_config , int ens_size , const char * job_script , int max_submit , int max_running , const char * lsf_queue_name , const stringlist_type * lsf_resource_request) {
  basic_queue_driver_type * driver = lsf_driver_alloc( lsf_queue_name , lsf_resource_request );
  if (site_config->job_queue != NULL)
    job_queue_free( site_config->job_queue );
  
  site_config->job_queue = job_queue_alloc(ens_size , max_running , max_submit , job_script , driver);
}



static void site_config_install_job_queue(site_config_type  * site_config , const config_type * config , int ens_size) {
  const char * queue_system = config_get(config , "QUEUE_SYSTEM");
  const char * job_script   = config_get(config , "JOB_SCRIPT");
  int   max_submit          = 3;
  
  if (strcmp(queue_system , "LSF") == 0) {
    const char * lsf_queue_name             = config_get(config , "LSF_QUEUE");
    stringlist_type * resource_request_list = config_alloc_complete_stringlist(config , "LSF_RESOURCES");
    int max_running;
    
    if (!util_sscanf_int(config_get(config , "MAX_RUNNING_LSF") , &max_running))
      util_abort("%s: internal error - \n",__func__);
    
    site_config_install_LSF_job_queue(site_config , ens_size , job_script , max_submit , max_running , lsf_queue_name , resource_request_list);
    stringlist_free( resource_request_list );
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


site_config_type * site_config_alloc(const config_type * config , int ens_size) {
  site_config_type * site_config = site_config_alloc_empty();
  site_config_install_job_queue(site_config , config , ens_size);
  site_config_install_joblist(site_config , config);
  site_config_set_image_viewer(site_config , config_get(config , "IMAGE_VIEWER"));
  return site_config;
}




/*****************************************************************/


enkf_fs_type * enkf_main_get_fs_ref(const enkf_main_type * ens) { return ens->fs; }


void enkf_main_insert_data_kw(enkf_main_type * enkf_main , int ens_size) {
  int ikw, iens , size;
  char ** data_kw_keys = enkf_config_alloc_data_kw_key_list(enkf_main->config , &size);

  if (size > 0) {
    for (iens = 0; iens < ens_size; iens++) 
      for (ikw = 0; ikw < size; ikw++) {
        const char * key   = data_kw_keys[ikw];
        const char * value = enkf_config_get_data_kw(enkf_main->config , key);
        enkf_state_set_data_kw(enkf_main->ensemble[iens] , key, value);
      }
    util_free_stringlist(data_kw_keys , size);
  }
}
             
lock_mode_type enkf_main_get_runlock_mode(const enkf_main_type * enkf_main) {
  return enkf_main->runlock_mode;
}

enkf_main_type * enkf_main_safe_cast( void * __enkf_main) {
  enkf_main_type * enkf_main = (enkf_main_type *) __enkf_main;
  if (enkf_main->__id != ENKF_MAIN_ID)
    util_abort("%s: run_time cast failed - aborting \n",__func__);
  return enkf_main;
}


/*
void enkf_main_clear_locks(enkf_main_type * enkf_main) {
  enkf_config_type * config = enkf_main->config;
  int ens_size = enkf_config_get_ens_size(config);
  int iens;
  for (iens = 0; iens < ens_size; iens++) {
    char * run_path = path_fmt_alloc_path(enkf_config_get_run_path_fmt(config), false , iens);
    enkf_state_steal_run_path_lock(enkf_main->ensemble[iens] , enkf_main->runlock_mode , enkf_main->lock_path , run_path);
    free(run_path);
  }
}
*/


/* 
   The runpath is pushed around all over the placce - UGGGGLY
*/

void enkf_main_interactive_set_runpath(enkf_main_type * enkf_main) {
  int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int  iens;
  char runpath_fmt[512];
  util_exit("%s: sorry currently broken\n",__func__);
  printf("Give new runpath (with %%d): ===> ");
  scanf("%s" , runpath_fmt);
  enkf_config_set_run_path(enkf_main->config , runpath_fmt);
  for (iens = 0; iens < ens_size; iens++) {
    char * run_path = path_fmt_alloc_path(enkf_config_get_run_path_fmt(enkf_main->config), false , iens);
    /*
      enkf_state_set_run_path(enkf_main->ensemble[iens] , enkf_main->runlock_mode , enkf_main->lock_path , run_path);
      
    */
    free(run_path);
  }
}


void enkf_main_interactive_set_runpath__(void * __enkf_main) {
  enkf_main_type * enkf_main = enkf_main_safe_cast(__enkf_main);
  enkf_main_interactive_set_runpath( enkf_main );
}



enkf_main_type * enkf_main_alloc(enkf_config_type * config, lock_mode_type lock_mode , const char * lock_path , enkf_fs_type *fs , job_queue_type * job_queue , ext_joblist_type * joblist) {
  int ens_size               = enkf_config_get_ens_size(config);

  enkf_main_type * enkf_main = util_malloc(sizeof *enkf_main, __func__);
  enkf_main->__id            = ENKF_MAIN_ID;
  enkf_main->config          = config;
  enkf_main->sched_file      = sched_file_alloc(enkf_config_get_start_date(config));

  {
    time_t start_date = -1;
    sched_file_parse(enkf_main->sched_file , start_date , enkf_config_get_schedule_src_file(config));
  }

  enkf_main->hist           = history_alloc_from_sched_file(enkf_main->sched_file);
  enkf_main->obs            = enkf_obs_fscanf_alloc(enkf_main->config , enkf_main->sched_file , enkf_main->hist);
  enkf_main->obs_data       = obs_data_alloc();
  enkf_main->fs             = fs;
  enkf_main->job_queue      = job_queue;
  enkf_main->meas_matrix    = meas_matrix_alloc(ens_size);
  enkf_main->ensemble       = malloc(ens_size * sizeof * enkf_main->ensemble);
  enkf_main->run_mode       = enkf_assimilation;
  enkf_main->runlock_mode   = lock_mode;
  enkf_main->lock_path      = util_alloc_string_copy(lock_path);
  util_make_path(enkf_main->lock_path);
  {
    int iens , keys , ik;
    char **keylist  = enkf_config_alloc_keylist(config , &keys);
    msg_type * msg  = msg_alloc("Initializing member: ");
    msg_show(msg);
    for (iens = 0; iens < ens_size; iens++) {
      msg_update_int(msg , "%03d" , iens);
      enkf_main->ensemble[iens] = enkf_state_alloc(config   , 
                                                   iens , 
                                                   enkf_main->runlock_mode , enkf_main->lock_path , enkf_config_iget_ecl_store(config , iens) , enkf_main->fs , 
                                                   joblist  , job_queue , 
                                                   enkf_main->sched_file ,
                                                   enkf_config_get_run_path_fmt(config),
                                                   enkf_config_get_eclbase_fmt(config),
                                                   meas_matrix_iget_vector(enkf_main->meas_matrix , iens),
                                                   enkf_main->obs);
      
    }
    msg_free(msg , true);

    msg  = msg_alloc("Adding key: ");
    msg_show(msg);
    for (ik = 0; ik < keys; ik++) {
      msg_update(msg , keylist[ik]);
      const enkf_config_node_type * config_node = enkf_config_get_node_ref(config , keylist[ik]);
      for (iens = 0; iens < ens_size; iens++) 
        enkf_state_add_node(enkf_main->ensemble[iens] , keylist[ik] , config_node);
    }
    msg_free(msg , true);


    util_free_stringlist(keylist , keys);
  }
  



  /*
   The user MUST specify an INIT_FILE, and for the first timestep the
   <INIT> tag in the data file will be replaced by an 

INCLDUE
   EQUIL_INIT_FILE

   statement. When considering the possibility of estimating EQUIL this
   require a real understanding of the treatment of paths:

   * If not estimating the EQUIL contacts, all members should use the
     same init_file. To ensure this the user must specify the ABSOLUTE
     PATH to a file containing initialization information.

   * If the user is estimating initial contacts, the INIT_FILE must
     point to the ecl_file of the EQUIL keyword, this must be a pure
     filename without any path component (as it will be generated by
     the EnKF program, and placed in the run_path directory). We could
     let the EnKF program use the ecl_file of the EQUIL keyword if it
     is present.

   Since correct behaviour here requires the "cooperation" of both the
   INIT_FILE keyword and the (possible) EQUIL keyword it is a bit
   tricky to enforce.

  */
  
  {
    const char * init_file   = enkf_config_get_init_file(config);
    if (init_file == NULL) 
      util_abort("%s: EQUIL_INIT_FILE is not set - must either use EQUIL_INIT_FILE in config_file or EQUIL keyword.",__func__);
    
    if (init_file != NULL) 
    {
      char * tmp_include     = util_alloc_joined_string((const char *[4]) {"  " , "'" , init_file , "' /"} , 4 , "");
      char * DATA_initialize = util_alloc_multiline_string((const char *[2]) {"INCLUDE" , tmp_include} , 2);

      enkf_main_set_data_kw(enkf_main , "INIT" , DATA_initialize);
      
      free(DATA_initialize);
      free(tmp_include);
    }
  }
  enkf_main_insert_data_kw(enkf_main , ens_size);
  return  enkf_main;

}


const enkf_config_type * enkf_main_get_config(const enkf_main_type * enkf_main) {
  return enkf_main->config;
}

const enkf_config_node_type * enkf_main_get_config_node(const enkf_main_type * enkf_main, const char * key) {
  enkf_config_type * config = enkf_main->config;
  return enkf_config_get_node_ref(config , key);
}



void enkf_main_set_data_kw(enkf_main_type * enkf_main , const char * new_kw , const char * value) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int iens;
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_set_data_kw(enkf_main->ensemble[iens] , new_kw , value);
}




void enkf_main_measure(enkf_main_type * enkf_main) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int iens;
  meas_matrix_reset(enkf_main->meas_matrix);
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_measure(enkf_main->ensemble[iens] , enkf_main->obs);
}



void enkf_main_free(enkf_main_type * enkf_main) {  
  enkf_obs_free(enkf_main->obs);
  {
    const int ens_size = enkf_config_get_ens_size(enkf_main->config);
    int i;
    for (i=0; i < ens_size; i++)
      enkf_state_free(enkf_main->ensemble[i]);
    free(enkf_main->ensemble);
  }
  free(enkf_main->lock_path);
  
  sched_file_free(enkf_main->sched_file);
  history_free(enkf_main->hist);
  meas_matrix_free(enkf_main->meas_matrix);
  obs_data_free(enkf_main->obs_data);
  enkf_config_free(enkf_main->config);
  free(enkf_main);
}


/*****************************************************************/


static const enkf_config_node_type * enkf_main_assert_obs(const enkf_main_type * enkf_main , const char *obs_key , enkf_impl_type impl_type) {
  if (enkf_config_has_key(enkf_main->config , obs_key)) {
    const enkf_config_node_type * config_node = enkf_config_get_node_ref(enkf_main->config , obs_key);
    if (enkf_config_node_get_impl_type(config_node) == impl_type) 
      return config_node;
    else {
      fprintf(stderr,"%s main object:%s exists - but it is not of correct type - aborting \n",__func__ , obs_key);
      abort();
    }
  } else {
    fprintf(stderr,"%s: main does not have key: %s - aborting \n",__func__ , obs_key);
    abort();
  }
}





/*****************************************************************/

void enkf_main_analysis(enkf_main_type * enkf_main) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int nrobs          = obs_data_get_nrobs(enkf_main->obs_data);
  
  if (nrobs > 0) {
    double * X = analysis_allocX(ens_size , obs_data_get_nrobs(enkf_main->obs_data) , enkf_main->meas_matrix , enkf_main->obs_data , false , true);
    free(X);
  }
}



void enkf_main_load_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state) {
  int iens;
  for (iens = 0; iens < enkf_config_get_ens_size(enkf_main->config); iens++) 
    enkf_state_fread(enkf_main->ensemble[iens] , mask , report_step , state);
}



void enkf_main_fwrite_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state) {
  int iens;
  for (iens = 0; iens < enkf_config_get_ens_size(enkf_main->config); iens++) 
    enkf_state_fwrite(enkf_main->ensemble[iens] , mask , report_step , state);
}




/**
   This function returns a (enkf_node_type ** ) pointer, which points
   to all the instances with the same keyword, i.e.

     enkf_main_get_node_ensemble(enkf_main , "PRESSURE");
  
   Will return an ensemble of pressure nodes. Observe that apart from
   the list of pointers, *now new storage* is allocated, all the
   pointers point in to the underlying enkf_node instances under the
   enkf_main / enkf_state objects. Consequently there is no designated
   free() function to match this, just free() the result.

   Example:

   enkf_node_type ** pressure_nodes = enkf_main_get_node_ensemble(enkf_main , "PRESSURE");
 
   Do something with the pressure nodes ... 

   free(pressure_nodes);

*/

enkf_node_type ** enkf_main_get_node_ensemble(const enkf_main_type * enkf_main , const char * key) {
  const int ens_size              = enkf_config_get_ens_size(enkf_main->config);
  enkf_node_type ** node_ensemble = util_malloc(ens_size * sizeof * node_ensemble , __func__ );
  int iens;

  for (iens = 0; iens < ens_size; iens++)
    node_ensemble[iens] = enkf_state_get_node(enkf_main->ensemble[iens] , key);
  
  return node_ensemble;
}

/*****************************************************************/

static void enkf_main_fprintf_results(const enkf_main_type * enkf_main , int report_step) {
  const int ens_size     = enkf_config_get_ens_size(enkf_main->config);
  int config_size;
  char ** key_list = enkf_config_alloc_keylist(enkf_main->config , &config_size);
  int ikw;

  for (ikw=0; ikw < config_size; ikw++) {
    const enkf_node_type * node = enkf_state_get_node(enkf_main->ensemble[0] , key_list[ikw]);
    if (enkf_node_has_func(node , ensemble_fprintf_results_func)) {
      enkf_node_type ** node_ensemble = enkf_main_get_node_ensemble(enkf_main , key_list[ikw]);
      char            * path          = enkf_config_alloc_result_path(enkf_main->config , report_step);
      enkf_node_ensemble_fprintf_results((const enkf_node_type **) node_ensemble , ens_size , report_step , path);
      free(path);
      free(node_ensemble);
    }
  }
  util_free_stringlist(key_list , config_size);
}


enkf_state_type * enkf_main_iget_state(const enkf_main_type * enkf_main , int iens) {
  return enkf_main->ensemble[iens];
}





/******************************************************************/

void enkf_main_run(enkf_main_type * enkf_main, const bool * iactive , int init_step , state_enum init_state , int step1 , int step2 , bool load_results , bool enkf_update, bool unlink_run_path , const stringlist_type * forward_model) {
  const int ens_size            = enkf_config_get_ens_size(enkf_main->config);
  int iens;
  
  if (enkf_update)
    load_results = true; 
  printf("Starting forward step: %d -> %d \n",step1,step2);
  enkf_obs_get_observations(enkf_main->obs , step2 , enkf_main->obs_data);
  meas_matrix_reset(enkf_main->meas_matrix);

  {
    pthread_t          queue_thread;
    void_arg_type * queue_args = void_arg_alloc2(void_pointer , int_value);
    void_arg_pack_ptr(queue_args , 0 , enkf_main->job_queue);
    void_arg_pack_int(queue_args , 1 , ens_size);
    
    pthread_create( &queue_thread , NULL , job_queue_run_jobs__ , queue_args);

    {
      thread_pool_type * submit_threads = thread_pool_alloc(4);
      for (iens = 0; iens < ens_size; iens++) {
        enkf_state_init_run(enkf_main->ensemble[iens] , iactive[iens] , init_step , init_state , step1 , step2 , load_results , unlink_run_path , forward_model);
        thread_pool_add_job(submit_threads , enkf_state_start_eclipse__ , enkf_main->ensemble[iens]);
      }
      thread_pool_join(submit_threads);  /* OK: All directories for ECLIPSE simulations are ready. */
      thread_pool_free(submit_threads);
    }
    
    {
      thread_pool_type * complete_threads = thread_pool_alloc(ens_size);
      for (iens = 0; iens < ens_size; iens++) 
        thread_pool_add_job(complete_threads , enkf_state_complete_eclipse__ , enkf_main->ensemble[iens]);
      
      thread_pool_join(complete_threads);        /* All jobs have completed and the results have been loaded back. */
      thread_pool_free(complete_threads);
    }

    pthread_join ( queue_thread , NULL );      /* The thread running the queue is complete.      */
    job_queue_finalize(enkf_main->job_queue);  /* Must *NOT* be called before all jobs are done. */               
    void_arg_free( queue_args );
  }
    
  {
    bool complete_OK = true;
    for (iens = 0; iens < ens_size; iens++) {
      if (! enkf_state_run_OK(enkf_main->ensemble[iens])) {
        if ( complete_OK ) {
          fprintf(stderr,"Some models failed to integrate from DATES %d -> %d:\n",step1 , step2);
          complete_OK = false;
        }
        fprintf(stderr,"** Error in: %s \n",enkf_state_get_run_path(enkf_main->ensemble[iens]));
      }
    }
    if (!complete_OK) 
      util_exit("The integration failed - check your forward model ...\n");
  }
  
  if (load_results) 
    enkf_main_load_ensemble(enkf_main , ecl_restart + ecl_summary + parameter , step2 , forecast);


  printf("Starter paa oppdatering \n");
  if (enkf_update) {
    double *X = analysis_allocX(ens_size , obs_data_get_nrobs(enkf_main->obs_data) , enkf_main->meas_matrix , enkf_main->obs_data , false , true);
    
    if (X != NULL) {
      /* 
         The number of doubles we ask for, to get the number of bytes
         you must multiply by eight.
         
         1024 * 1024 * 128 => 1GB of memory
      */
      size_t double_size = 1024*1024*256; /* 2GB */
      
      serial_vector_type * serial_vector = serial_vector_alloc( double_size , ens_size );  /* DANGER DANGER DANGER - might go fatally low on memory when the serial_vector is held. */
      enkf_ensemble_update(enkf_main->ensemble , ens_size , serial_vector , X);   
      serial_vector_free(serial_vector);
      
      free(X);
    }
  }
    
  printf("---------------------------------\n");
  if (enkf_update) {
    enkf_main_fwrite_ensemble(enkf_main , parameter + ecl_restart + ecl_summary , step2 , analyzed);
    enkf_main_fprintf_results(enkf_main , step2);
  }
  printf("%s: ferdig med step: %d \n" , __func__,step2);
}



const sched_file_type * enkf_main_get_sched_file(const enkf_main_type * enkf_main) { return enkf_main->sched_file; }




void enkf_main_bootstrap(const char * _site_config, const char * _model_config) {
  char * cwd               = util_alloc_cwd();
  const char * site_config = getenv("ENKF_SITE_CONFIG");
  char       * model_config;
  
  printf("-----------------------------------------------------------------\n");
  printf("Test code: start \n");
  if (site_config == NULL)
    site_config = _site_config;
  
  if (site_config == NULL) 
    util_exit("%s: main enkf_config file is not set. Use environment variable \"ENKF_SITE_CONFIG\" - or recompile - aborting.\n",__func__);
  
  {
    char * path;
    char * base;
    char * ext;
    util_alloc_file_components(_model_config , &path , &base , &ext);
    if (path != NULL) {
      if (chdir(path) != 0) 
        util_abort("%s: failed to change directory to: %s : %s \n",__func__ , path , strerror(errno));

      printf("Changing to directory ...................: %s \n",path);
      if (ext != NULL) {
        model_config = util_alloc_joined_string((const char *[3]) {base , "." , ext} , 3 , "");
        free(base);
      } else 
        model_config = base;
      free(ext);
      free(path);
    } else
      model_config = util_alloc_string_copy(_model_config);
  }  

  if (!util_file_exists(site_config)) util_exit("%s: can not locate site configuration file:%s \n",__func__ , site_config);
  if (!util_file_exists(model_config)) util_exit("%s: can not locate user configuration file:%s \n",__func__ , model_config);
  {
    config_type * config = config_alloc();
    config_item_type * item;

    /*****************************************************************/
    /* config_add_item():                                            */
    /*                                                               */
    /*  1. boolean - required                                        */
    /*  2. boolean - append?                                         */
    /*****************************************************************/

    /*****************************************************************/
    /** Keywords expected normally found in site_config */
    item = config_add_item(config , "QUEUE_SYSTEM" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    {
      stringlist_type * selection  = stringlist_alloc_argv_ref( (const char *[3]) {"LSF" , "LOCAL" , "RSH"} , 3);
      stringlist_type * lsf_dep    = stringlist_alloc_argv_ref( (const char *[3]) {"LSF_RESOURCES" , "LSF_QUEUE" , "MAX_RUNNING_LSF"}   , 3);
      stringlist_type * rsh_dep    = stringlist_alloc_argv_ref( (const char *[3]) {"RSH_HOST_LIST" , "RSH_COMMAND" , "MAX_RUNNING_RSH"} , 2);
      stringlist_type * local_dep  = stringlist_alloc_argv_ref( (const char *[1]) {"MAX_RUNNING_LOCAL"}   , 1);

      config_item_set_selection_set( item , selection);
      config_item_set_required_children_on_value( item , "LSF"   , lsf_dep);
      config_item_set_required_children_on_value( item , "RSH"   , rsh_dep);
      config_item_set_required_children_on_value( item , "LOCAL" , local_dep);

      stringlist_free(selection);
      stringlist_free(lsf_dep);
      stringlist_free(rsh_dep);
      stringlist_free(local_dep);
    }
    
                                                
    /* These must be set IFF QUEUE_SYSTEM == LSF */
    config_add_item(config , "LSF_RESOURCES" , false , true);
    item = config_add_item(config , "LSF_QUEUE"     , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);

    item = config_add_item(config , "MAX_RUNNING_LSF" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});


    /* These must be set IFF QUEUE_SYSTEM == RSH */
    config_add_item(config , "RSH_HOST_LIST" , false , false);
    item = config_add_item(config , "RSH_COMMAND" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXECUTABLE});
    item = config_add_item(config , "MAX_RUNNING_RSH" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});
    
    
    /* These must be set IFF QUEUE_SYSTEM == LOCAL */
    item = config_add_item(config , "MAX_RUNNING_LOCAL" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});


    item = config_add_item(config , "JOB_SCRIPT" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});

    item = config_add_item(config , "IMAGE_VIEWER" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});

    
    item = config_add_item(config , "INSTALL_JOB" , true , true);
    config_item_set_argc_minmax(item , 2 , 2 , (const config_item_types [2]) {CONFIG_STRING , CONFIG_EXISTING_FILE});

    
    /*****************************************************************/
    /* Required keywords from the ordinary model_config file */
    item = config_add_item(config , "SIZE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});
    
    item = config_add_item(config , "GRID" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});

    item = config_add_item(config , "RUNPATH" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);

    item = config_add_item(config , "ENSPATH" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);

    item = config_add_item(config , "ECLBASE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    
    item = config_add_item(config , "SCHEDULE_FILE" , true , false);
    config_item_set_argc_minmax(item , 1 , 2 , (const config_item_types [2]) {CONFIG_EXISTING_FILE , CONFIG_STRING});

    item = config_add_item(config , "START_TIME" , true , false);
    /*config_item_set_argc_minmax(item , 1 , 2 , (const config_item_types [2]) {CONFIG_EXISTING_FILE , CONFIG_STRING});*/

    item = config_add_item(config , "DATA_FILE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});

    item = config_add_item(config , "INIT_FILE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});
    
    item = config_add_item(config , "FORWARD_MODEL" , true , false);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);
    /*****************************************************************/
    /* Optional keywords from the model config file */
    item = config_add_item(config , "DATA_KW" , false , true);
    config_item_set_argc_minmax(item , 2 , 2 , NULL);

    item = config_add_item(config , "KEEP_RUNPATH" , false , false);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);

    item = config_add_item(config , "ADD_STATIC_KW" , false , false);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);

    item = config_add_item(config , "RESULT_PATH"  , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);

    item = config_add_item(config , "OBS_CONFIG"  , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) { CONFIG_EXISTING_FILE});

    
    /*****************************************************************/
    /* Keywords for the estimation                                   */
    item = config_add_item(config , "MULTZ" , false , true);
    config_item_set_argc_minmax(item , 3 , 3 ,  (const config_item_types [3]) { CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});
    
    item = config_add_item(config , "MULTFLT" , false , true);
    config_item_set_argc_minmax(item , 3 , 3 ,  (const config_item_types [3]) { CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});
    
    item = config_add_item(config , "EQUIL" , false , true);
    config_item_set_argc_minmax(item , 3 , 3 ,  (const config_item_types [3]) { CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});

    item = config_add_item(config , "GEN_KW" , false , true);
    config_item_set_argc_minmax(item , 4 , 4 ,  (const config_item_types [4]) { CONFIG_STRING , CONFIG_EXISTING_FILE , CONFIG_STRING , CONFIG_EXISTING_FILE});

    item = config_add_item(config , "GEN_PARAM" , false , true);
    config_item_set_argc_minmax(item , 3 , 4 ,  (const config_item_types [4]) { CONFIG_STRING , CONFIG_STRING , CONFIG_STRING , CONFIG_EXISTING_FILE});

    item = config_add_item(config , "WELL" , false , true);
    config_item_set_argc_minmax(item , 2 , -1 ,  NULL);

    item = config_add_item(config , "SUMMARY" , false , true);
    config_item_set_argc_minmax(item , 2 , -1 ,  NULL);

    item = config_add_item(config , "FIELD" , false , true);
    config_item_set_argc_minmax(item , 2 , -1 ,  NULL);
    

    
    config_parse(config , site_config  , "--" , "INCLUDE" , false , false);
    config_parse(config , model_config , "--" , "INCLUDE" , false , true);

    {
      int ens_size = strtol(config_get(config , "SIZE") , NULL , 10);
      site_config_type * site_config = site_config_alloc(config , ens_size);
      
      site_config_free( site_config );
    }
    
    printf("Skal kalle config_free .. \n");
    config_free(config);
    printf("Config_free() complete \n");
  }
  free(model_config);
  chdir( cwd ); /* Noninvasive in test mode ... */
  free( cwd );  
  printf("Test code: end \n");
  printf("-----------------------------------------------------------------\n");
}
    
    
    



  
