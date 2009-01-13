#include <errno.h>
#include <string.h>
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
#include <field_config.h>
#include <equil_config.h>
#include <multz_config.h>
#include <multflt_config.h>
#include <thread_pool.h>
#include <obs_data.h>
#include <history.h>
#include <meas_matrix.h>
#include <enkf_state.h>  
#include <analysis.h>
#include <enkf_obs.h>
#include <sched_file.h>
#include <enkf_fs.h>
#include <arg_pack.h>
#include <gen_kw_config.h>
#include <gen_data_config.h>
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
#include <active_config.h>
#include <field_config.h>
#include <ecl_static_kw.h>
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
       enkf_main object is taken. In the former case it is no way to
       change global behaviour by modifying the enkf_main objects.
       
       In the enkf_state object the fields of the member_config,
       ecl_config, site_config and ensemble_config objects are mixed
       and matched into other small holding objects defined in
       enkf_state.c.

*/

#define ENKF_MAIN_ID 8301

struct enkf_main_struct {
  int                    __type_id;        /* Used for type-checking run-time casts. */
  ensemble_config_type * ensemble_config;  /* The config objects for the various enkf nodes.*/ 
  model_config_type    * model_config;
  ecl_config_type      * ecl_config;
  site_config_type     * site_config;
  analysis_config_type * analysis_config;
  enkf_obs_type        * obs;
  obs_data_type        * obs_data;          /* Should ideally contain the hist object. */
  meas_matrix_type     * meas_forecast;     /* Should have both forecast and analyzed measurements ... */
  meas_matrix_type     * meas_analyzed;
  enkf_state_type     ** ensemble;
}; 




/*****************************************************************/




SAFE_CAST(enkf_main , ENKF_MAIN_ID)
     /*
       enkf_main_type * enkf_main_safe_cast( void * __enkf_main) {
       enkf_main_type * enkf_main = (enkf_main_type *) __enkf_main;
       if (enkf_main->__type_id != ENKF_MAIN_ID)
       util_abort("%s: run_time cast failed - aborting \n",__func__);
       return enkf_main;
       }
     */


const ensemble_config_type * enkf_main_get_ensemble_config(const enkf_main_type * enkf_main) {
  return enkf_main->ensemble_config;
}

model_config_type * enkf_main_get_model_config( const enkf_main_type * enkf_main ) {
  return enkf_main->model_config;
}


const enkf_sched_type * enkf_main_get_enkf_sched(const enkf_main_type * enkf_main) {
  return model_config_get_enkf_sched(enkf_main->model_config);
}


enkf_fs_type * enkf_main_get_fs(const enkf_main_type * enkf_main) {
  return model_config_get_fs( enkf_main->model_config );
}


enkf_obs_type * enkf_main_get_obs(const enkf_main_type * enkf_main) {
  return enkf_main->obs;
}




void enkf_main_measure(enkf_main_type * enkf_main , int report_step , state_enum state) {
  const int ens_size = ensemble_config_get_size(enkf_main->ensemble_config);
  meas_matrix_type * meas_matrix;

  if (state == forecast) 
    meas_matrix = enkf_main->meas_forecast;
  else
    meas_matrix = enkf_main->meas_analyzed;
  
  meas_matrix_reset(meas_matrix);
  enkf_obs_measure_on_ensemble( enkf_main->obs , enkf_main_get_fs(enkf_main) , report_step , state , ens_size , (const enkf_state_type **) enkf_main->ensemble , meas_matrix);
}



void enkf_main_free(enkf_main_type * enkf_main) {  
  enkf_obs_free(enkf_main->obs);
  obs_data_free(enkf_main->obs_data);
  {
    const int ens_size = ensemble_config_get_size(enkf_main->ensemble_config);
    int i;
    for (i=0; i < ens_size; i++)
      enkf_state_free(enkf_main->ensemble[i]);
    free(enkf_main->ensemble);
  }

  analysis_config_free(enkf_main->analysis_config);
  ecl_config_free(enkf_main->ecl_config);
  model_config_free( enkf_main->model_config);
  site_config_free( enkf_main->site_config);
  ensemble_config_free( enkf_main->ensemble_config );
  meas_matrix_free( enkf_main->meas_forecast );
  meas_matrix_free( enkf_main->meas_analyzed );
  free(enkf_main);
}



/*****************************************************************/



static void enkf_main_load_sub_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state, int iens1 , int iens2) {
  int iens;
  for (iens = iens1; iens < iens2; iens++)
    enkf_state_fread(enkf_main->ensemble[iens] , mask , report_step , state );
}


static void * enkf_main_load_sub_ensemble__(void * __arg) {
  arg_pack_type * arg_pack   = arg_pack_safe_cast(__arg);
  enkf_main_type * enkf_main = arg_pack_iget_ptr(arg_pack , 0);
  int mask                   = arg_pack_iget_int(arg_pack , 1);
  int report_step            = arg_pack_iget_int(arg_pack , 2);
  state_enum state           = arg_pack_iget_int(arg_pack , 3);
  int iens1                  = arg_pack_iget_int(arg_pack , 4);
  int iens2                  = arg_pack_iget_int(arg_pack , 5);

  enkf_main_load_sub_ensemble(enkf_main , mask , report_step , state , iens1 , iens2);
  return NULL;
}



void enkf_main_load_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state) {
  const   int cpu_threads = 4;
  int     sub_ens_size    = ensemble_config_get_size(enkf_main->ensemble_config) / cpu_threads;
  int     icpu;
  thread_pool_type * tp          = thread_pool_alloc( cpu_threads );
  arg_pack_type ** arg_pack_list = util_malloc( cpu_threads * sizeof * arg_pack_list , __func__);
  
  for (icpu = 0; icpu < cpu_threads; icpu++) {
    arg_pack_type * arg = arg_pack_alloc();
    arg_pack_append_ptr(arg , enkf_main);
    arg_pack_append_int(arg , mask);
    arg_pack_append_int(arg , report_step);
    arg_pack_append_int(arg , state);
    
    {
      int iens1 =  icpu * sub_ens_size;
      int iens2 = iens1 + sub_ens_size;
      
      if (icpu == (cpu_threads - 1))
	iens2 = ensemble_config_get_size(enkf_main->ensemble_config);

      arg_pack_append_int(arg ,  iens1);
      arg_pack_append_int(arg ,  iens2);
    }
    arg_pack_list[icpu] = arg;
    arg_pack_lock( arg );
    thread_pool_add_job( tp , enkf_main_load_sub_ensemble__ , arg);
  }
  thread_pool_join( tp );
  thread_pool_free( tp );

  for (icpu = 0; icpu < cpu_threads; icpu++) 
    arg_pack_free( arg_pack_list[icpu] );
  free(arg_pack_list);
}





static void enkf_main_fwrite_sub_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state, int iens1 , int iens2) {
  int iens;
  for (iens = iens1; iens < iens2; iens++)
    enkf_state_fwrite(enkf_main->ensemble[iens] , mask , report_step , state);
}


static void * enkf_main_fwrite_sub_ensemble__(void *__arg) {
  arg_pack_type * arg_pack   = arg_pack_safe_cast(__arg);
  enkf_main_type * enkf_main = arg_pack_iget_ptr(arg_pack , 0);
  int mask                   = arg_pack_iget_int(arg_pack , 1);
  int report_step            = arg_pack_iget_int(arg_pack , 2);
  state_enum state           = arg_pack_iget_int(arg_pack , 3);
  int iens1                  = arg_pack_iget_int(arg_pack , 4);
  int iens2                  = arg_pack_iget_int(arg_pack , 5);

  enkf_main_fwrite_sub_ensemble(enkf_main , mask , report_step , state , iens1 , iens2);
  return NULL;
}


void enkf_main_fwrite_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state) {
  const   int cpu_threads = 4;
  int     sub_ens_size    = ensemble_config_get_size(enkf_main->ensemble_config) / cpu_threads;
  int     icpu;
  thread_pool_type * tp = thread_pool_alloc( cpu_threads );
  arg_pack_type ** arg_pack_list = util_malloc( cpu_threads * sizeof * arg_pack_list , __func__);
  
  for (icpu = 0; icpu < cpu_threads; icpu++) {
    arg_pack_type * arg = arg_pack_alloc();
    arg_pack_append_ptr(arg , enkf_main);
    arg_pack_append_int(arg , mask);
    arg_pack_append_int(arg , report_step);
    arg_pack_append_int(arg , state);
    
    {
      int iens1 =  icpu * sub_ens_size;
      int iens2 = iens1 + sub_ens_size;
      
      if (icpu == (cpu_threads - 1))
	iens2 = ensemble_config_get_size(enkf_main->ensemble_config);

      arg_pack_append_int(arg , iens1);
      arg_pack_append_int(arg , iens2);
    }
    arg_pack_list[icpu] = arg;
    arg_pack_lock( arg );
    thread_pool_add_job( tp , enkf_main_fwrite_sub_ensemble__ , arg);
  }
  thread_pool_join( tp );
  thread_pool_free( tp );

  for (icpu = 0; icpu < cpu_threads; icpu++) 
    arg_pack_free( arg_pack_list[icpu]);
  free(arg_pack_list);
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
  const int ens_size  = ensemble_config_get_size(enkf_main->ensemble_config);
  int config_size;
  char ** key_list    = ensemble_config_alloc_keylist(enkf_main->ensemble_config , &config_size);
  int ikw;

  for (ikw=0; ikw < config_size; ikw++) {
    /* 
       Unfortunately we can have config_nodes without actual nodes (in
       the case) of STATIC. They are not printed anyway.
    */
    if (enkf_config_node_get_impl_type(ensemble_config_get_node(enkf_main->ensemble_config , key_list[ikw])) != STATIC) {
      const enkf_node_type * node = enkf_state_get_node(enkf_main->ensemble[0] , key_list[ikw]);
      if (enkf_node_has_func(node , ensemble_fprintf_results_func)) {
	enkf_node_type ** node_ensemble = enkf_main_get_node_ensemble(enkf_main , key_list[ikw]);
	char            * path          = model_config_alloc_result_path(enkf_main->model_config , report_step);
	
	enkf_node_ensemble_fprintf_results((const enkf_node_type **) node_ensemble , ens_size , report_step , path);
	free(path);
	free(node_ensemble);
      }
    }
  }
  util_free_stringlist(key_list , config_size);
}


enkf_state_type * enkf_main_iget_state(const enkf_main_type * enkf_main , int iens) {
  return enkf_main->ensemble[iens];
}





/******************************************************************/


void enkf_main_analysis_update(enkf_main_type * enkf_main , int report_step) {
  const int ens_size            = ensemble_config_get_size(enkf_main->ensemble_config);
  double *X;
  enkf_obs_get_observations(enkf_main->obs , report_step , enkf_main->obs_data);
  enkf_main_measure(enkf_main , report_step , forecast);
  X = analysis_allocX(ens_size , obs_data_get_nrobs(enkf_main->obs_data) , enkf_main->meas_forecast , enkf_main->obs_data , false , true , enkf_main->analysis_config);
  if (X != NULL) {
    /* 
       The number of doubles we ask for, to get the number of bytes
       you must multiply by eight.
       
       1024 * 1024 * 128 => 1GB of memory
    */
    size_t double_size = 1024*1024*256; /* 2GB */
    
    /* DANGER DANGER DANGER - might go fatally low on memory when the serial_vector is held. */
    printf("Updating: ...... "); fflush(stdout);
    serial_vector_type * serial_vector = serial_vector_alloc( double_size , ens_size );  
    enkf_ensemble_update(enkf_main->ensemble , ens_size , serial_vector , X);   
    serial_vector_free(serial_vector);
    printf("\n");
    free(X);
  }

  /* This will write analyzed results to disk anyway - maybe a bit wastefull . */
  printf("Saving: ........ "); fflush(stdout);
  enkf_main_fwrite_ensemble(enkf_main , dynamic_state + dynamic_result + parameter , report_step , analyzed);
  printf("\n");
  enkf_main_measure(enkf_main , report_step , analyzed);
  
  /** Printing update info after analysis. */
  {
    double *meanS , *stdS;
    meas_matrix_allocS_stats(enkf_main->meas_analyzed , &meanS , &stdS);
    obs_data_fprintf(enkf_main->obs_data , stdout , meanS , stdS);
    free(meanS);
    free(stdS);
  }
  
  enkf_main_fprintf_results(enkf_main , report_step);
}



void enkf_main_run_step(enkf_main_type * enkf_main, run_mode_type run_mode , const bool * iactive , int init_step , state_enum init_state , int step1 , int step2 , bool enkf_update , const stringlist_type * forward_model) {
  const int ens_size            = ensemble_config_get_size(enkf_main->ensemble_config);
  int   job_size;

  int iens;
  
  printf("Starting forward step: %d -> %d\n",step1 , step2);
  site_config_update_lsf_request(enkf_main->site_config , forward_model);

  job_size = 0;
  for (iens = 0; iens < ens_size; iens++)
    if (iactive[iens]) job_size++;

  {
    pthread_t          queue_thread;
    job_queue_type * job_queue = site_config_get_job_queue(enkf_main->site_config);
    arg_pack_type * queue_args = arg_pack_alloc();
    arg_pack_append_ptr(queue_args , job_queue);
    arg_pack_append_int(queue_args , job_size);
    arg_pack_lock( queue_args );
    pthread_create( &queue_thread , NULL , job_queue_run_jobs__ , queue_args);

    {
      thread_pool_type * submit_threads = thread_pool_alloc(4);
      for (iens = 0; iens < ens_size; iens++) {
        enkf_state_init_run(enkf_main->ensemble[iens] , run_mode , iactive[iens] , init_step , init_state , step1 , step2, forward_model);
        thread_pool_add_job(submit_threads , enkf_state_start_eclipse__ , enkf_main->ensemble[iens]);
      }
      thread_pool_join(submit_threads);  /* OK: All directories for ECLIPSE simulations are ready. */
      thread_pool_free(submit_threads);
    }
    
    {
      thread_pool_type * complete_threads = thread_pool_alloc(ens_size);
      for (iens = 0; iens < ens_size; iens++) 
        thread_pool_add_job(complete_threads , enkf_state_complete_eclipse__ , enkf_main->ensemble[iens]);
      
      thread_pool_join(complete_threads);      /* All jobs have completed and the results have been loaded back. */
      thread_pool_free(complete_threads);
    }
    pthread_join ( queue_thread , NULL );      /* The thread running the queue is complete.      */
    job_queue_finalize(job_queue);             /* Must *NOT* be called before all jobs are done. */               
    arg_pack_free( queue_args );
    enkf_main_del_unused_static( enkf_main , step2 );
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
  
  if (enkf_update)
    enkf_main_analysis_update(enkf_main , step2);
  
  printf("%s: ferdig med step: %d \n" , __func__,step2);
}




void * enkf_main_get_enkf_config_node_type(const ensemble_config_type * ensemble_config, const char * key){
  enkf_config_node_type * config_node_type = ensemble_config_get_node(ensemble_config, key);
  return enkf_config_node_get_ref(config_node_type);
}




void enkf_main_set_field_config_iactive(const ensemble_config_type * ensemble_config, int local_step){
  
  /*
    bool * test_field_iactive;
    const gen_data_config_type * gen_data_config = enkf_main_get_enkf_config_node_type(ensemble_config,"AI"); 
    int num_param = gen_data_config_get_num_param(gen_data_config,local_step);
    int * i= gen_data_config_get_param_index_i(gen_data_config,local_step);
    int * j= gen_data_config_get_param_index_j(gen_data_config,local_step);
    int * k= gen_data_config_get_param_index_k(gen_data_config,local_step);
    printf("num_param:%d, i:%d,j:%d,k:%d", num_param,i[0],j[0],k[0]);
  */

  field_config_type * poro_config = enkf_main_get_enkf_config_node_type(ensemble_config,"PORO");
  field_config_activate(poro_config , partly_active , NULL);
}




const char * enkf_main_get_image_viewer(const enkf_main_type * enkf_main) {
  return site_config_get_image_viewer(enkf_main->site_config);
}




void enkf_main_run(enkf_main_type * enkf_main , const bool * iactive ,  int start_report , state_enum __init_state) {
  bool analyzed_start;
  bool prev_enkf_on;
  const enkf_sched_type * enkf_sched = enkf_main_get_enkf_sched(enkf_main);
  const int num_nodes            = enkf_sched_get_num_nodes(enkf_sched);
  const int schedule_num_reports = enkf_sched_get_schedule_num_reports(enkf_sched);
  const int start_inode          = enkf_sched_get_node_index(enkf_sched , start_report);
  int inode;
  
  if (__init_state == analyzed)
    analyzed_start = true;
  else
    analyzed_start = false;

  
  prev_enkf_on = analyzed_start;
  for (inode = start_inode; inode < num_nodes; inode++) {
    const enkf_sched_node_type * node = enkf_sched_iget_node(enkf_sched , inode);
    state_enum init_state;
    int 	   init_step;
    int 	   report_step1;
    int 	   report_step2;
    int 	   report_stride;
    int 	   report_step;
    int 	   next_report_step;
    bool enkf_on;
    stringlist_type * forward_model;
    
    enkf_sched_node_get_data(node , &report_step1 , &report_step2 , &report_stride , &enkf_on , &forward_model);
    if (inode == start_inode)
      report_step = start_report;
    else
      report_step = report_step1;
    do {
      next_report_step = util_int_min(schedule_num_reports , util_int_min(report_step + report_stride , report_step2));
      init_step = report_step;
      if (prev_enkf_on)
	init_state = analyzed;
      else
	init_state = forecast;
      
      enkf_main_run_step(enkf_main , enkf_assimilation , iactive , init_step , init_state , report_step , next_report_step , enkf_on , forward_model);
      report_step  = next_report_step;
      prev_enkf_on = enkf_on;
    } while (next_report_step < report_step2);
  }
}


/**
   This function boots everything needed for running a EnKF
   application. Very briefly it can be summarized as follows:

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
  enkf_main->__type_id  = ENKF_MAIN_ID;
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
    /*  1. boolean - required?                                       */
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
      stringlist_type * lsf_dep    = stringlist_alloc_argv_ref( (const char *[2]) {"LSF_QUEUE" , "MAX_RUNNING_LSF"}   , 2);
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
    config_add_item(config , "LSF_RESOURCES" , false , false);
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

    item = config_add_item(config , "KEEP_RUNPATH" , false , false);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);

    item = config_add_item(config , "ADD_STATIC_KW" , false , true);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);
    
    item = config_add_item(config , "RESULT_PATH"  , true , false);
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
    
    item = config_add_item(config , "PLOT_PATH" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    config_set_arg(config , "PLOT_PATH" , 1 , (const char *[1]) { DEFAULT_PLOT_PATH });

    /*****************************************************************/
    /* Keywords for the analysis - all optional. */
    item = config_add_item(config , "ENKF_MODE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    config_item_set_common_selection_set(item , 2 , (const char *[2]) {"STANDARD" , "SQRT"});
    config_set_arg(config , "ENKF_MODE" , 1 , (const char *[1]) { DEFAULT_ENKF_MODE });

    item = config_add_item(config , "ENKF_ALPHA" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types[1]) { CONFIG_FLOAT });
    config_set_arg(config , "ENKF_ALPHA" , 1 , (const char *[1]) { DEFAULT_ENKF_ALPHA });

    item = config_add_item(config , "ENKF_TRUNCATION" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types[1]) { CONFIG_FLOAT });
    config_set_arg(config , "ENKF_TRUNCATION" , 1 , (const char *[1]) { DEFAULT_ENKF_TRUNCATION });

    
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
    enkf_main->analysis_config = analysis_config_alloc(config);


    /*****************************************************************/
    {
      const int ens_size = ensemble_config_get_size( enkf_main->ensemble_config );
      bool * keep_runpath = util_malloc( sizeof * keep_runpath * ens_size , __func__);
      int i;
      for (i = 0; i < ens_size; i++) 
	keep_runpath[i] = true;

      if (config_has_set_item(config , "KEEP_RUNPATH")) {
	char * keep_runpath_string = config_indexed_alloc_joined_string(config , "KEEP_RUNPATH" , "" , i);
	util_sscanf_active_range(keep_runpath_string , ens_size - 1 , keep_runpath);
	free( keep_runpath_string );
      }
    

      if (config_has_set_item(config , "ADD_STATIC_KW")) {
	for (i=0; i < config_get_occurences(config , "ADD_STATIC_KW"); i++) {
	  const stringlist_type * static_kw_list = config_iget_stringlist_ref(config , "ADD_STATIC_KW" , i);
	  int k;
	  for (k = 0; k < stringlist_get_size(static_kw_list); k++)
	    ecl_config_add_static_kw(enkf_main->ecl_config , stringlist_iget( static_kw_list , k));
	}
      }
      
      {
	const char * obs_config_file;
	if (config_has_set_item(config , "OBS_CONFIG"))
	  obs_config_file = config_get(config  , "OBS_CONFIG");
	else
	  obs_config_file = NULL;
	
	enkf_main->meas_forecast   = meas_matrix_alloc(ensemble_config_get_size(enkf_main->ensemble_config));
	enkf_main->meas_analyzed   = meas_matrix_alloc(ensemble_config_get_size(enkf_main->ensemble_config));
	enkf_main->obs             = enkf_obs_fscanf_alloc(obs_config_file , model_config_get_history(enkf_main->model_config) , enkf_main->ensemble_config);
	enkf_main->obs_data        = obs_data_alloc();
	{
	  stringlist_type * summary_vars = enkf_obs_alloc_summary_vars(enkf_main->obs);
	  for (i=0; i < stringlist_get_size( summary_vars ); i++) 
	    ensemble_config_ensure_summary( enkf_main->ensemble_config , stringlist_iget( summary_vars , i));
	  stringlist_free( summary_vars );
	}
      }

      /******************************************************************/
      /* Adding inverse observation keys, and config_nodes to the obs_vectors. */
      {
	hash_type * map = enkf_obs_alloc_summary_map(enkf_main->obs);
	const char * obs_key = hash_iter_get_first_key(map);
	while (obs_key  != NULL) {
	  const char * state_kw = hash_get(map , obs_key);
	  ensemble_config_add_obs_key(enkf_main->ensemble_config , state_kw , obs_key);
	  
	  {
	    obs_vector_type * obs_vector = enkf_obs_get_vector(enkf_main->obs , obs_key);
	    obs_vector_set_config_node( obs_vector , ensemble_config_get_node( enkf_main->ensemble_config , state_kw ));
	  }
	  obs_key = hash_iter_get_next_key( map );
	}
	hash_free(map);
      }
      

      
      /*****************************************************************/
      /* Adding ensemble members */
      
      {
	hash_type * data_kw = config_alloc_hash(config , "DATA_KW");
	int iens , keys , ik;
	char **keylist  = ensemble_config_alloc_keylist(enkf_main->ensemble_config , &keys);
	msg_type * msg  = msg_alloc("Initializing member: ");
	msg_show(msg);
	enkf_main->ensemble = util_malloc(ensemble_config_get_size(enkf_main->ensemble_config) * sizeof * enkf_main->ensemble , __func__);
	for (iens = 0; iens < ens_size; iens++) {
	  msg_update_int(msg , "%03d" , iens);
	  enkf_main->ensemble[iens] = enkf_state_alloc(iens,
						       keep_runpath[iens],
						       enkf_main->model_config   , 
						       enkf_main->ensemble_config,
						       enkf_main->site_config    , 
						       enkf_main->ecl_config     ,
						       data_kw);
	  
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
      free(keep_runpath);
    }
    config_free(config);
  }
  free(model_config);
  return enkf_main;
}
    


/**
   First deleting all the nodes - then the configuration.
*/

void enkf_main_del_node(enkf_main_type * enkf_main , const char * key) {
  const int ens_size = ensemble_config_get_size(enkf_main->ensemble_config);
  int iens;
  for (iens = 0; iens < ens_size; iens++) 
    enkf_state_del_node(enkf_main->ensemble[iens] , key);
  ensemble_config_del_node(enkf_main->ensemble_config , key);
}


/*
  This function will discard all unused static nodes. 
*/
void enkf_main_del_unused_static(enkf_main_type * enkf_main , int report_step) {
  int config_size;
  char ** key_list    = ensemble_config_alloc_keylist(enkf_main->ensemble_config , &config_size);
  int ikw;
  
  for (ikw=0; ikw < config_size; ikw++) {
    if (enkf_config_node_get_impl_type(ensemble_config_get_node(enkf_main->ensemble_config , key_list[ikw])) == STATIC) {
      enkf_node_type * node = enkf_state_get_node(enkf_main->ensemble[0] , key_list[ikw]);
      
      if (enkf_node_get_impl_type(node) == STATIC) {
	ecl_static_kw_type * ecl_static_kw = enkf_node_value_ptr(node);
	if (ecl_static_kw_get_report_step(ecl_static_kw) != report_step) /* This kw has not been loaded for this report step. */
	  enkf_main_del_node(enkf_main , key_list[ikw] );
      }
    }
  }
}




  
