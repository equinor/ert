#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <util.h>
#include <hash.h>
#include <multz_config.h>
#include <enkf_config_node.h>
#include <ecl_util.h>
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
#include <ecl_io_config.h>
#include <ecl_config.h>
#include <ensemble_config.h>
#include <model_config.h>
#include <site_config.h>
#include "enkf_defaults.h"







/**
   This object should contain **everything** needed to run a enkf
   simulation. A way to wrap up all available information/state and
   pass it around. An attempt has been made to collect various pieces
   of related information together in a couple of objects
   (model_config, ecl_config, site_config and ensemble_config). When
   it comes to these holding objects the following should be observed:

    1. It not always obvious where a piece of information should be
       stored, i.e. the grid is a property of the model, however it is
       an eclipse grid, and hence also belongs to eclipse
       configuration?? [In this case ecl_config wins out.]

    2. The information stored in these objects is typically passed on
       to the enkf_state object, where it is used. 

    3. At enkf_state level it is not really consequent - in some cases
       the enkf_state object takes a scalar copy (i.e. keep_runpath),
       and in other cases only a pointer down to the underlying
       enkf_main object is taken.
       In the former case it is no way to change global behaviour by
       modifying the enkf_main objects. 
       
    4. In the enkf_state object the fields of the member_config,
       ecl_config, site_config and ensemble_config objects are mixed
       and matched into other small holding objects defined in
       enkf_state.c.

*/

#define ENKF_MAIN_ID 8301

struct enkf_main_struct {
  int                    __id;             /* Used for type-checking run-time casts. */
  ensemble_config_type * ensemble_config;  /* The config objects for the various enkf nodes.*/ 
  model_config_type    * model_config;
  ecl_config_type      * ecl_config;
  site_config_type     * site_config;
  enkf_obs_type        * obs;
  obs_data_type        * obs_data;      /* Should ideally contain the hist object. */
  meas_matrix_type     * meas_matrix;
  enkf_state_type     ** ensemble;
}; 




/*****************************************************************/





enkf_main_type * enkf_main_safe_cast( void * __enkf_main) {
  enkf_main_type * enkf_main = (enkf_main_type *) __enkf_main;
  if (enkf_main->__id != ENKF_MAIN_ID)
    util_abort("%s: run_time cast failed - aborting \n",__func__);
  return enkf_main;
}


const ensemble_config_type * enkf_main_get_ensemble_config(const enkf_main_type * enkf_main) {
  return enkf_main->ensemble_config;
}


const enkf_sched_type * enkf_main_get_enkf_sched(const enkf_main_type * enkf_main) {
  return model_config_get_enkf_sched(enkf_main->model_config);
}


enkf_fs_type * enkf_main_get_fs(const enkf_main_type * enkf_main) {
  return model_config_get_fs( enkf_main->model_config );
}






void enkf_main_measure(enkf_main_type * enkf_main) {
  const int ens_size = ensemble_config_get_size(enkf_main->ensemble_config);
  int iens;
  meas_matrix_reset(enkf_main->meas_matrix);
  for (iens = 0; iens < ens_size; iens++)
     enkf_state_measure(enkf_main->ensemble[iens] , enkf_main->obs);
}



void enkf_main_free(enkf_main_type * enkf_main) {  
  enkf_obs_free(enkf_main->obs);
{
const int ens_size = ensemble_config_get_size(enkf_main->ensemble_config);
int i;
for (i=0; i < ens_size; i++)
      enkf_state_free(enkf_main->ensemble[i]);
      free(enkf_main->ensemble);
      }

ecl_config_free(enkf_main->ecl_config);
model_config_free( enkf_main->model_config);
site_config_free( enkf_main->site_config);
  ensemble_config_free( enkf_main->ensemble_config );
  free(enkf_main);
}



/*****************************************************************/


void enkf_main_analysis(enkf_main_type * enkf_main) {
  const int ens_size = ensemble_config_get_size(enkf_main->ensemble_config);
  int nrobs          = obs_data_get_nrobs(enkf_main->obs_data);
  
  if (nrobs > 0) {
    double * X = analysis_allocX(ens_size , obs_data_get_nrobs(enkf_main->obs_data) , enkf_main->meas_matrix , enkf_main->obs_data , false , true);
    free(X);
  }
}



void enkf_main_load_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state) {
  int iens;
  for (iens = 0; iens < ensemble_config_get_size(enkf_main->ensemble_config); iens++) 
    enkf_state_fread(enkf_main->ensemble[iens] , mask , report_step , state);
}



void enkf_main_fwrite_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state) {
  int iens;
  for (iens = 0; iens < ensemble_config_get_size(enkf_main->ensemble_config); iens++) 
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
  const int ens_size              = ensemble_config_get_size(enkf_main->ensemble_config);
  enkf_node_type ** node_ensemble = util_malloc(ens_size * sizeof * node_ensemble , __func__ );
  int iens;

  for (iens = 0; iens < ens_size; iens++)
    node_ensemble[iens] = enkf_state_get_node(enkf_main->ensemble[iens] , key);
  
  return node_ensemble;
}

/*****************************************************************/

static void enkf_main_fprintf_results(const enkf_main_type * enkf_main , int report_step) {
  const int ens_size     = ensemble_config_get_size(enkf_main->ensemble_config);
  int config_size;
  char ** key_list = ensemble_config_alloc_keylist(enkf_main->ensemble_config , &config_size);
  int ikw;

  for (ikw=0; ikw < config_size; ikw++) {
    const enkf_node_type * node = enkf_state_get_node(enkf_main->ensemble[0] , key_list[ikw]);
    if (enkf_node_has_func(node , ensemble_fprintf_results_func)) {
      enkf_node_type ** node_ensemble = enkf_main_get_node_ensemble(enkf_main , key_list[ikw]);
      char            * path          = model_config_alloc_result_path(enkf_main->model_config , report_step);
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
  const int ens_size            = ensemble_config_get_size(enkf_main->ensemble_config);
  int iens;
  
  if (enkf_update)
    load_results = true; 
  printf("Starting forward step: %d -> %d \n",step1,step2);
  enkf_obs_get_observations(enkf_main->obs , step2 , enkf_main->obs_data);
  meas_matrix_reset(enkf_main->meas_matrix);

  {
    pthread_t          queue_thread;
job_queue_type * job_queue = site_config_get_job_queue(enkf_main->site_config);
    void_arg_type * queue_args = void_arg_alloc2(void_pointer , int_value);
    void_arg_pack_ptr(queue_args , 0 , job_queue);
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
    job_queue_finalize(job_queue);             /* Must *NOT* be called before all jobs are done. */               
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



/**
   This function boots more or less everything needed for running a
   EnKF application. Very briefly it can be summarized as follows:

    1. A large config object is initalized with all the possible
       keywords we are looking for.

    2. All the config files are parsed in one go.

    3. The various objects are build up by reading from the config
       object.

    4. The resulting enkf_main object contains *EVERYTHING*
       (whoaha...)
*/



enkf_main_type * enkf_main_bootstrap(const char * _site_config, const char * _model_config) {
  enkf_main_type * enkf_main = util_malloc(sizeof *enkf_main, __func__);
  
  const char * site_config = getenv("ENKF_SITE_CONFIG");
  char       * model_config;
  
  if (site_config == NULL)
    site_config = _site_config;
  
  if (site_config == NULL) 
    util_exit("%s: main enkf_config file is not set. Use environment variable \"ENKF_SITE_CONFIG\" - or recompile - aborting.\n",__func__);
  enkf_main->__id            = ENKF_MAIN_ID;
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
    item = config_add_item(config , "MAX_SUBMIT" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});
    config_set_arg(config , "MAX_SUBMIT" , 1 , (const char *[1]) { DEFAULT_MAX_SUBMIT});
    

    item = config_add_item(config , "QUEUE_SYSTEM" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    {
      stringlist_type * lsf_dep    = stringlist_alloc_argv_ref( (const char *[3]) {"LSF_RESOURCES" , "LSF_QUEUE" , "MAX_RUNNING_LSF"}   , 3);
      stringlist_type * rsh_dep    = stringlist_alloc_argv_ref( (const char *[3]) {"RSH_HOST_LIST" , "RSH_COMMAND" , "MAX_RUNNING_RSH"} , 2);
      stringlist_type * local_dep  = stringlist_alloc_argv_ref( (const char *[1]) {"MAX_RUNNING_LOCAL"}   , 1);

      config_item_set_common_selection_set( item , 3 , (const char *[3]) {"LSF" , "LOCAL" , "RSH"});
      config_item_set_required_children_on_value( item , "LSF"   , lsf_dep);
      config_item_set_required_children_on_value( item , "RSH"   , rsh_dep);
      config_item_set_required_children_on_value( item , "LOCAL" , local_dep);
      
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
    item = config_add_item(config , "NUM_REALIZATIONS" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});
    config_add_alias(config , "NUM_REALIZATIONS" , "SIZE");
    config_install_message(config , "SIZE" , "** Warning: \'SIZE\' is depreceated - use \'NUM_REALIZATIONS\' instead.");
    

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

    item = config_add_item(config , "EQUIL_INIT_FILE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});
    
    item = config_add_item(config , "FORWARD_MODEL" , true , false);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);
    /*****************************************************************/
    /* Optional keywords from the model config file */
    item = config_add_item(config , "DATA_KW" , false , true);
    config_item_set_argc_minmax(item , 2 , 2 , NULL);

    item = config_add_item(config , "KEEP_RUNPATH" , false , true);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);

    item = config_add_item(config , "ADD_STATIC_KW" , false , true);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);
    
    item = config_add_item(config , "RESULT_PATH"  , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);

    item = config_add_item(config , "OBS_CONFIG"  , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) { CONFIG_EXISTING_FILE});
    
    item = config_add_item(config , "REFCASE" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) { CONFIG_EXISTING_FILE});

    item = config_add_item(config , "ENKF_SCHED_FILE" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) { CONFIG_EXISTING_FILE});

    item = config_add_item(config , "HISTORY_SOURCE" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    {
      stringlist_type * refcase_dep = stringlist_alloc_argv_ref( (const char *[1]) {"REFCASE"} , 1);

      config_item_set_common_selection_set(item , 3 , (const char *[3]) {"SCHEDULE" , "REFCASE_OBSERVED" , "REFCASE_HISTORY"});
      config_item_set_required_children_on_value(item , "REFCASE_OBSERVED" , refcase_dep);
      config_item_set_required_children_on_value(item , "REFCASE_HISTORY"  , refcase_dep);

      stringlist_free(refcase_dep);
    }
      config_set_arg(config , "HISTORY_SOURCE" , 1 , (const char *[1]) { DEFAULT_HISTORY_SOURCE });
    
    /*****************************************************************/
    /* Keywords for the estimation                                   */
    ensemble_config_add_config_items(config); 
    
    
    config_parse(config , site_config  , "--" , "INCLUDE" , false , false);
    config_parse(config , model_config , "--" , "INCLUDE" , false , true);
    /*****************************************************************/
    /* OK - now we have parsed everything - and we are ready to start
       populating the enkf_main object. 
    */
    
     
    enkf_main->ecl_config      = ecl_config_alloc(config);
    enkf_main->ensemble_config = ensemble_config_alloc( config , ecl_config_get_grid( enkf_main->ecl_config ));
    enkf_main->site_config     = site_config_alloc(config , ensemble_config_get_size( enkf_main->ensemble_config ));
    enkf_main->model_config    = model_config_alloc(config , site_config_get_installed_jobs(enkf_main->site_config) , ecl_config_get_sched_file(enkf_main->ecl_config));


    {
      const char * obs_config_file;
      if (config_has_set_item(config , "OBS_CONFIG"))
	obs_config_file = config_get(config  , "OBS_CONFIG");
      else
	obs_config_file = NULL;
      
      enkf_main->meas_matrix     = meas_matrix_alloc(ensemble_config_get_size(enkf_main->ensemble_config));
      enkf_main->obs             = enkf_obs_fscanf_alloc(obs_config_file , enkf_main->ensemble_config ,  ecl_config_get_sched_file(enkf_main->ecl_config) , model_config_get_history(enkf_main->model_config));
      enkf_main->obs_data        = obs_data_alloc();
    }
    
    /*****************************************************************/
    /* Adding ensemble members */
    {
      const int ens_size = ensemble_config_get_size(enkf_main->ensemble_config);
      hash_type * data_kw = config_alloc_hash(config , "DATA_KW");
      int iens , keys , ik;
      char **keylist  = ensemble_config_alloc_keylist(enkf_main->ensemble_config , &keys);
      msg_type * msg  = msg_alloc("Initializing member: ");
      msg_show(msg);
      enkf_main->ensemble = util_malloc(ensemble_config_get_size(enkf_main->ensemble_config) * sizeof * enkf_main->ensemble , __func__);
      for (iens = 0; iens < ens_size; iens++) {
	msg_update_int(msg , "%03d" , iens);
	enkf_main->ensemble[iens] = enkf_state_alloc(iens,
						     enkf_main->model_config   , 
						     enkf_main->ensemble_config,
						     enkf_main->site_config    , 
						     enkf_main->ecl_config     ,
						     data_kw,
						     meas_matrix_iget_vector(enkf_main->meas_matrix , iens),
						     enkf_main->obs);
	
      }
      msg_free(msg , true);
      
      msg  = msg_alloc("Adding key: ");
      msg_show(msg);
      for (ik = 0; ik < keys; ik++) {
        msg_update(msg , keylist[ik]);
	const enkf_config_node_type * config_node = ensemble_config_get_node(enkf_main->ensemble_config , keylist[ik]);
	for (iens = 0; iens < ens_size; iens++) 
	  enkf_state_add_node(enkf_main->ensemble[iens] , keylist[ik] , config_node);
      }
      msg_free(msg , true);
      hash_free(data_kw);
      util_free_stringlist(keylist , keys);
    }
    
    config_free(config);
  }
  free(model_config);
  return enkf_main;
}
    
    


    



  
