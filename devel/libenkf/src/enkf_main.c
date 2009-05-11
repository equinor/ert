#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <util.h>
#include <hash.h>
#include <enkf_config_node.h>
#include <ecl_util.h>
#include <path_fmt.h>
#include <ecl_static_kw_config.h>
#include <enkf_types.h>
#include <field_config.h>
#include <thread_pool.h>
#include <obs_data.h>
#include <history.h>
#include <meas_matrix.h>
#include <enkf_state.h>  
#include <old_analysis.h>
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
#include <forward_model.h>
#include <enkf_analysis.h>
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
  enkf_state_type     ** ensemble;
}; 




/*****************************************************************/

void enkf_main_init_internalization( enkf_main_type *  , run_mode_type  );

/*****************************************************************/

SAFE_CAST(enkf_main , ENKF_MAIN_ID)


ensemble_config_type * enkf_main_get_ensemble_config(const enkf_main_type * enkf_main) {
  return enkf_main->ensemble_config;
}

model_config_type * enkf_main_get_model_config( const enkf_main_type * enkf_main ) {
  return enkf_main->model_config;
}


int enkf_main_get_history_length( const enkf_main_type * enkf_main) {
  return model_config_get_last_history_restart( enkf_main->model_config);
}

int enkf_main_get_total_length( const enkf_main_type * enkf_main) {
  return model_config_get_abs_last_restart( enkf_main->model_config );
}

bool enkf_main_has_prediction( const enkf_main_type * enkf_main ) {
  return model_config_has_prediction( enkf_main->model_config );
}



//const enkf_sched_type * enkf_main_get_enkf_sched(const enkf_main_type * enkf_main) {
//return model_config_get_enkf_sched(enkf_main->model_config);
//}


enkf_fs_type * enkf_main_get_fs(const enkf_main_type * enkf_main) {
  return model_config_get_fs( enkf_main->model_config );
}


enkf_obs_type * enkf_main_get_obs(const enkf_main_type * enkf_main) {
  return enkf_main->obs;
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

  analysis_config_free(enkf_main->analysis_config);
  ecl_config_free(enkf_main->ecl_config);
  model_config_free( enkf_main->model_config);
  site_config_free( enkf_main->site_config);
  ensemble_config_free( enkf_main->ensemble_config );
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
  stringlist_type * key_list    = ensemble_config_alloc_keylist(enkf_main->ensemble_config);
  int config_size = stringlist_get_size(key_list);

  for (int ikw=0; ikw < config_size; ikw++) {
    /* 
       Unfortunately we can have config_nodes without actual nodes (in
       the case) of STATIC. They are not printed anyway.
    */
    const char * key = stringlist_iget(key_list, ikw);
    if (enkf_config_node_get_impl_type(ensemble_config_get_node(enkf_main->ensemble_config , key)) != STATIC) {
      const enkf_node_type * node = enkf_state_get_node(enkf_main->ensemble[0] , key);
      if (enkf_node_has_func(node , ensemble_fprintf_results_func)) {
	enkf_node_type ** node_ensemble = enkf_main_get_node_ensemble(enkf_main , key);
	char            * path          = model_config_alloc_result_path(enkf_main->model_config , report_step);
	
	enkf_node_ensemble_fprintf_results((const enkf_node_type **) node_ensemble , ens_size , report_step , path);
	free(path);
	free(node_ensemble);
      }
    }
  }
  stringlist_free(key_list);
}


enkf_state_type * enkf_main_iget_state(const enkf_main_type * enkf_main , int iens) {
  return enkf_main->ensemble[iens];
}





void enkf_main_analysis_update(enkf_main_type * enkf_main , int step1 , int step2) {
  /* 
     If include_internal_observations is true all observations in the
     time interval [step1+1,step2] will be used, otherwise only the
     last observation at step2 will be used.
  */
  bool include_internal_observations = analysis_config_merge_observations( enkf_main->analysis_config );
  const int ens_size                 = ensemble_config_get_size(enkf_main->ensemble_config);
  double *X;
  int start_step , end_step;
  
  /* Observe that end_step is inclusive. */
  if (include_internal_observations) {
    start_step = step1 + 1;
    end_step   = step2;
  } else {
    start_step = step2;
    end_step   = step2;
  }
    
  
  {
    /*
      Observations and measurements are collected in these temporary
      structures. obs_data is a precursor for the 'd' vector, and
      meas_forecast is a precursor for the 'S' matrix'

      The reason for gong via these temporary structures is to support
      deactivating observations which can not be used in the update
      process.
    */
    
    
    obs_data_type    * obs_data      = obs_data_alloc();
    meas_matrix_type * meas_forecast = meas_matrix_alloc( ens_size );
    meas_matrix_type * meas_analyzed = meas_matrix_alloc( ens_size );

    for(int report_step = start_step; report_step <= end_step; report_step++)  {
      printf("Fetching simulated responses and observations for step %i.\n", report_step); 
      enkf_obs_get_obs_and_measure(enkf_main->obs, enkf_main_get_fs(enkf_main), report_step, forecast, ens_size, 
				   (const enkf_state_type **) enkf_main->ensemble, meas_forecast, obs_data);
    }
    X = old_analysis_allocX(ens_size , obs_data_get_nrobs(obs_data) , meas_forecast , obs_data , false , true , enkf_main->analysis_config);
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
    enkf_main_fwrite_ensemble(enkf_main , DYNAMIC_STATE + DYNAMIC_RESULT + PARAMETER , step2 , analyzed);
    printf("\n");
    
    /*
      Collect observations and simulated responses for all steps after step1 up to and
      including step2.
    */
    
    
    /** Printing update info after analysis - does not work with multi step updates.*/
    if (start_step == end_step) {
      obs_data_reset(obs_data);
      
      for(int report_step = start_step; report_step <= end_step; report_step++)  {
	printf("Fetching simulated responses and observations for step %i.\n", report_step);
	enkf_obs_get_obs_and_measure(enkf_main->obs, enkf_main_get_fs(enkf_main), report_step, analyzed, ens_size, (const enkf_state_type **) enkf_main->ensemble, meas_analyzed, obs_data);
      }
      {
	double *meanS , *stdS;
	meas_matrix_allocS_stats(meas_analyzed , &meanS , &stdS);
	obs_data_fprintf(obs_data , stdout , meanS , stdS);
	free(meanS);
	free(stdS);
      }
      enkf_main_fprintf_results(enkf_main , step2);
    }
    obs_data_free( obs_data );
    meas_matrix_free( meas_forecast );
    meas_matrix_free( meas_analyzed );
  }
}



void enkf_main_run_step(enkf_main_type * enkf_main, 
			run_mode_type    run_mode , 
			const bool * iactive      , 
			int init_step             , 
			state_enum init_state     , 
			int step1                 , 
			int step2                 , 
			bool enkf_update          , 
			forward_model_type * forward_model) {  /* The forward model will be != NULL ONLY if it is different from the default forward model. */
  const int ens_size    = ensemble_config_get_size(enkf_main->ensemble_config);
  int   job_size;

  int iens;
  
  printf("Starting forward step: %d -> %d\n",step1 , step2);
  
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
        enkf_state_init_run(enkf_main->ensemble[iens] , run_mode , iactive[iens] , init_step , init_state , step1 , step2 , forward_model);
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
    enkf_main_analysis_update(enkf_main , step1 , step2);
  
  printf("%s: ferdig med step: %d \n" , __func__,step2);
}




void * enkf_main_get_enkf_config_node_type(const ensemble_config_type * ensemble_config, const char * key){
  enkf_config_node_type * config_node_type = ensemble_config_get_node(ensemble_config, key);
  return enkf_config_node_get_ref(config_node_type);
}



void enkf_main_set_field_config_iactive(const ensemble_config_type * ensemble_config, int local_step){
  field_config_type * poro_config = enkf_main_get_enkf_config_node_type(ensemble_config,"PORO");
  field_config_activate(poro_config , PARTLY_ACTIVE , NULL);
}




const char * enkf_main_get_image_viewer(const enkf_main_type * enkf_main) {
  return site_config_get_image_viewer(enkf_main->site_config);
}

const char * enkf_main_get_image_type(const enkf_main_type * enkf_main) {
  return site_config_get_image_type(enkf_main->site_config);
}


/**
   This function will initialize the necessary enkf_main structures
   before a run. Currently this means:

     1. Set the enkf_sched instance - either by loading from file or
        by using the default.
	 
     2. Set up the configuration of what should be internalized.

*/



void enkf_main_init_run( enkf_main_type * enkf_main, run_mode_type run_mode) {
  const ext_joblist_type * joblist = site_config_get_installed_jobs( enkf_main->site_config);

  model_config_set_enkf_sched( enkf_main->model_config , joblist , run_mode);
  enkf_main_init_internalization(enkf_main , run_mode);

}


/**
   The main RUN function - will run both enkf assimilations and experiments.
*/
void enkf_main_run(enkf_main_type * enkf_main , 
		   run_mode_type    run_mode  , 
		   const bool     * iactive   ,  
		   int parameter_init_report  , 
		   int start_report           , 
		   state_enum start_state) {
  
  enkf_main_init_run( enkf_main , run_mode);  
  {
    enkf_fs_type * fs = enkf_main_get_fs(enkf_main);
    if (run_mode == ENKF_ASSIMILATION) {
      if (enkf_fs_rw_equal(fs)) {
	bool analyzed_start = false;
	bool prev_enkf_on;
	const enkf_sched_type * enkf_sched = model_config_get_enkf_sched(enkf_main->model_config);
	const int num_nodes                = enkf_sched_get_num_nodes(enkf_sched);
	const int start_inode              = enkf_sched_get_node_index(enkf_sched , start_report);
	int inode;
	
	if (start_state == analyzed)
	  analyzed_start = true;
	else if (start_state == forecast)
	  analyzed_start = false;
	else
	  util_abort("%s: internal error - start_state must be analyzed | forecast \n",__func__);
	
	prev_enkf_on = analyzed_start;
	for (inode = start_inode; inode < num_nodes; inode++) {
	  const enkf_sched_node_type * node = enkf_sched_iget_node(enkf_sched , inode);
	  state_enum init_state;
	  int 	   report_step1;
	  int 	   report_step2;
	  bool enkf_on;
	  forward_model_type * forward_model;
	  
	  enkf_sched_node_get_data(node , &report_step1 , &report_step2 , &enkf_on , &forward_model);
	  if (inode == start_inode) 
	    report_step1 = start_report;  /* If we are restarting from somewhere. */
	  
	  if (prev_enkf_on)
	    init_state = analyzed;
	  else
	    init_state = forecast;
	  
	  enkf_main_run_step(enkf_main , ENKF_ASSIMILATION , iactive , report_step1 , init_state , report_step1 , report_step2 , enkf_on , forward_model);
	  prev_enkf_on = enkf_on;
	}
      } else
	fprintf(stderr , "\n** Error: when running EnKF read and write directories must be equal.\n\n");
    } else {
      /* It is an experiment */
      const enkf_sched_type * enkf_sched = model_config_get_enkf_sched(enkf_main->model_config);
      const int last_report              = enkf_sched_get_last_report(enkf_sched);
      if (run_mode == ENSEMBLE_EXPERIMENT) {
	/* No possibility to use funky forward model */
	enkf_main_run_step(enkf_main , ENSEMBLE_EXPERIMENT , iactive , parameter_init_report , start_state , start_report , last_report , false , NULL );
      } else if (run_mode == SCREENING_EXPERIMENT) {
	enkf_main_run_step(enkf_main , SCREENING_EXPERIMENT , iactive , parameter_init_report , start_state , start_report , last_report , false , NULL );
      } else 
	util_abort("%s: internal error - invalid value for run_mode:%d \n",__func__ , run_mode);
    }
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
    
    /* 
       You can set environment variables which will be applied to the
       run-time environment. Can unfortunately not use constructions
       like PATH=$PATH:/some/new/path, use the UPDATE_PATH function instead.
    */
    item = config_add_item(config , "SETENV" , false , true);
    config_item_set_argc_minmax(item , 2 , 2 , NULL);

    /**
       UPDATE_PATH   LD_LIBRARY_PATH   /path/to/some/funky/lib

       Will prepend "/path/to/some/funky/lib" at the front of LD_LIBRARY_PATH.
    */
    item = config_add_item(config , "UPDATE_PATH" , false , true);
    config_item_set_argc_minmax(item , 2 , 2 , NULL);
    


    
    /* These must be set IFF QUEUE_SYSTEM == LSF */
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

    item = config_add_item(config , "IMAGE_TYPE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    config_item_set_common_selection_set( item , 3 , (const char *[3]) {"png" , "jpg" , "psc"});
    config_set_arg(config , "IMAGE_TYPE" , 1 , (const char *[1]) { DEFAULT_IMAGE_TYPE });

    item = config_add_item(config , "PLOT_PATH" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    config_set_arg(config , "PLOT_PATH" , 1 , (const char *[1]) { DEFAULT_PLOT_PATH });

    
    item = config_add_item(config , "INSTALL_JOB" , true , true);
    config_item_set_argc_minmax(item , 2 , 2 , (const config_item_types [2]) {CONFIG_STRING , CONFIG_EXISTING_FILE});

    
    /*****************************************************************/
    /* Required keywords from the ordinary model_config file */
    item = config_add_item(config , "NUM_REALIZATIONS" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_INT});
    config_add_alias(config , "NUM_REALIZATIONS" , "SIZE");
    config_add_alias(config , "NUM_REALIZATIONS" , "NUM_REALISATIONS");
    config_install_message(config , "SIZE" , "** Warning: \'SIZE\' is depreceated - use \'NUM_REALIZATIONS\' instead.");
    

    item = config_add_item(config , "GRID" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});

    item = config_add_item(config , "ECLBASE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    
    item = config_add_item(config , "SCHEDULE_FILE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});
    
    item = config_add_item(config , "SCHEDULE_PREDICTION_FILE" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    
    item = config_add_item(config , "DATA_FILE" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});
    
    item = config_add_item(config , "INIT_SECTION" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_FILE});
    config_add_alias(config , "INIT_SECTION" , "EQUIL_INIT_FILE"); 
    
    /*****************************************************************/
    /* Optional keywords from the model config file */

    item = config_add_item(config , "SCHEDULE_PREDICTION_FILE" , false , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types [1]) {CONFIG_EXISTING_FILE});

    item = config_add_item(config , "RUNPATH" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    config_set_arg(config , "RUNPATH" , 1 , (const char *[1]) { DEFAULT_RUNPATH });

    item = config_add_item(config , "ENSPATH" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    config_set_arg(config , "ENSPATH" , 1 , (const char *[1]) { DEFAULT_ENSPATH });

    item = config_add_item(config , "FORWARD_MODEL" , true , false);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);
    config_set_arg(config , "FORWARD_MODEL" , 1 , (const char *[1]) { DEFAULT_FORWARD_MODEL });

    item = config_add_item(config , "DATA_KW" , false , true);
    config_item_set_argc_minmax(item , 2 , 2 , NULL);

    item = config_add_item(config , "KEEP_RUNPATH" , false , false);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);

    item = config_add_item(config , "DELETE_RUNPATH" , false , false);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);

    item = config_add_item(config , "ADD_STATIC_KW" , false , true);
    config_item_set_argc_minmax(item , 1 , -1 , NULL);
    
    item = config_add_item(config , "RESULT_PATH"  , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , NULL);
    config_set_arg(config , "RESULT_PATH" , 1 , (const char *[1]) { DEFAULT_RESULT_PATH });

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
      
      config_item_set_common_selection_set(item , 3 , (const char *[3]) {"SCHEDULE" , "REFCASE_SIMULATED" , "REFCASE_HISTORY"});
      config_item_set_required_children_on_value(item , "REFCASE_SIMULATED" , refcase_dep);
      config_item_set_required_children_on_value(item , "REFCASE_HISTORY"  , refcase_dep);

      stringlist_free(refcase_dep);
    }
    config_set_arg(config , "HISTORY_SOURCE" , 1 , (const char *[1]) { DEFAULT_HISTORY_SOURCE });
    
    
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

    item = config_add_item(config , "ENKF_MERGE_OBSERVATIONS" , true , false);
    config_item_set_argc_minmax(item , 1 , 1 , (const config_item_types[1]) { CONFIG_BOOLEAN });
    config_set_arg(config , "ENKF_MERGE_OBSERVATIONS" , 1 , (const char *[1]) { DEFAULT_MERGE_OBSERVATIONS });
        
    /*****************************************************************/
    /* Keywords for the estimation                                   */
    ensemble_config_add_config_items(config); 
    
    
    config_parse(config , site_config  , "--" , "INCLUDE" , "DEFINE" , enkf_util_alloc_tagged_string , false , false);
    config_parse(config , model_config , "--" , "INCLUDE" , "DEFINE" , enkf_util_alloc_tagged_string , false , true);
    /*****************************************************************/
    /* OK - now we have parsed everything - and we are ready to start
       populating the enkf_main object. 
    */
    
    enkf_main->analysis_config = analysis_config_alloc(config);
    {
      bool use_lsf;
      enkf_main->ecl_config      = ecl_config_alloc( config );
      enkf_main->ensemble_config = ensemble_config_alloc( config , ecl_config_get_grid( enkf_main->ecl_config ));
      enkf_main->site_config     = site_config_alloc(config , ensemble_config_get_size( enkf_main->ensemble_config ) , &use_lsf);
      enkf_main->model_config    = model_config_alloc(config , 
						      site_config_get_installed_jobs(enkf_main->site_config) , 
						      ecl_config_get_last_history_restart( enkf_main->ecl_config ), 
						      ecl_config_get_sched_file(enkf_main->ecl_config) , use_lsf);
    }


    /*****************************************************************/
    /**
       To keep or not to keep the runpath directories? The problem is
       that the default behavior is different depending on the run_mode:

       enkf_mode: In this case the default behaviour is to delete the
       runpath directories. You can explicitly say that you want to
       keep runpath directories with the KEEP_RUNPATH
       directive. 

       experiments: In this case the default is to keep the runpath
       directories around, but you can explicitly say that you
       want to remove the directories by using the DELETE_RUNPATH
       option.

       The final decision is performed in enkf_state().
    */
    {
      const int ens_size = ensemble_config_get_size( enkf_main->ensemble_config );
      keep_runpath_type * keep_runpath = util_malloc( ens_size * sizeof * keep_runpath , __func__);
      
      int i;
      for (i = 0; i < ens_size; i++) 
	keep_runpath[i] = DEFAULT_KEEP;

      if (config_has_set_item(config , "KEEP_RUNPATH")) {
	char * keep_runpath_string = config_indexed_alloc_joined_string(config , "KEEP_RUNPATH" , "" , i);
	bool * flag                = util_malloc( sizeof * flag * ens_size , __func__);

	util_sscanf_active_range(keep_runpath_string , ens_size - 1 , flag);
	for (i = 0; i < ens_size; i++) {
	  if (flag[i]) 
	    keep_runpath[i] = EXPLICIT_KEEP;
	}
	
	free( flag );
	free( keep_runpath_string );
      }

      if (config_has_set_item(config , "DELETE_RUNPATH")) {
	char * delete_runpath_string = config_indexed_alloc_joined_string(config , "DELETE_RUNPATH" , "" , i);
	bool * flag                = util_malloc( sizeof * flag * ens_size , __func__);
	
	util_sscanf_active_range(delete_runpath_string , ens_size - 1 , flag);
	for (i = 0; i < ens_size; i++) {
	  if (flag[i]) {
	    if (keep_runpath[i] == EXPLICIT_KEEP)
	      util_abort("%s: Inconsistent use of KEEP_RUNPATH / DELETE_RUNPATH - trying to both keep and delete member:%d \n",__func__ , i);
	    keep_runpath[i] = EXPLICIT_DELETE;
	  }
	}

	free(flag );
	free( delete_runpath_string );
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
	
	enkf_main->obs             = enkf_obs_fscanf_alloc(obs_config_file , model_config_get_history(enkf_main->model_config) , enkf_main->ensemble_config);
      }

      /******************************************************************/
      /* Adding inverse observation keys, and config_nodes to the obs_vectors. */
      {
	hash_type      * map = enkf_obs_alloc_summary_map(enkf_main->obs);
        hash_iter_type * iter = hash_iter_alloc(map);
	const char * obs_key = hash_iter_get_next_key(iter);
	while (obs_key  != NULL) {
	  const char * state_kw = hash_get(map , obs_key);
	  ensemble_config_add_obs_key(enkf_main->ensemble_config , state_kw , obs_key);
	  obs_key = hash_iter_get_next_key(iter);
	}
        hash_iter_free(iter);
	hash_free(map);
      }
      

      
      /*****************************************************************/
      /* Adding ensemble members */
      
      {
	hash_type       * data_kw  =  config_alloc_hash(config , "DATA_KW");
	stringlist_type * keylist  = ensemble_config_alloc_keylist(enkf_main->ensemble_config);

        int keys = stringlist_get_size(keylist);

	msg_type * msg  = msg_alloc("Initializing member: ");
	msg_show(msg);
	enkf_main->ensemble = util_malloc(ensemble_config_get_size(enkf_main->ensemble_config) * sizeof * enkf_main->ensemble , __func__);
	for (int iens = 0; iens < ens_size; iens++) {
	  msg_update_int(msg , "%03d" , iens);
	  enkf_main->ensemble[iens] = enkf_state_alloc(iens,
						       keep_runpath[iens],
						       enkf_main->model_config   , 
						       enkf_main->ensemble_config,
						       enkf_main->site_config    , 
						       enkf_main->ecl_config     ,
						       data_kw,
						       model_config_get_std_forward_model(enkf_main->model_config));

	  /** This is the time we tell the model config object about our 'maximum report step' - possibly including predictions. */
	  model_config_update_last_restart(enkf_main->model_config , enkf_state_get_last_restart_nr( enkf_main->ensemble[iens] ));
	}
	msg_free(msg , true);
	
	msg  = msg_alloc("Adding key: ");
	msg_show(msg);
	for (int ik = 0; ik < keys; ik++) {
          const char * key = stringlist_iget(keylist, ik);
	  msg_update(msg , key);
	  const enkf_config_node_type * config_node = ensemble_config_get_node(enkf_main->ensemble_config , key);
	  for (int iens = 0; iens < ens_size; iens++) 
	    enkf_state_add_node(enkf_main->ensemble[iens] , key , config_node);
	}
	msg_free(msg , true);
	hash_free(data_kw);
        stringlist_free(keylist);
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




enkf_state_type ** enkf_main_get_ensemble( enkf_main_type * enkf_main) {
  return enkf_main->ensemble;
}



/**
   In this function we initialize the the variables which control
   which nodes are internalized (i.e. loaded from the forward
   simulation and stored in the enkf_fs 'database'). The system is
   based on two-levels:

   * Should we store the state? This is goverened by the variable
     model_config->internalize_state. If this is true we will
     internalize all nodes which have enkf_var_type = {dynamic_state ,
     static_state}. In the same way the variable
     model_config->internalize_results governs whether the dynamic
     results (i.e. summary variables in ECLIPSE speak) should be
     internalized.

   * In addition we have fine-grained control in the enkf_config_node
     objects where we can explicitly say that, altough we do not want
     to internalize the full state, we want to internalize e.g. the
     pressure field.
 
   * All decisions on internalization are based on a per report step
     basis.
   
   The user-space API for manipulating this is (extremely)
   limited. What is implemented here is the following:

     1. We internalize the initial dynamic state.

     2. For all the end-points in the current enkf_sched instance we
        internalize the state.

     3. store_results is set to true for all report steps irrespective
        of run_mode.

     4. We iterate over all the observations, and ensure that the
        observed nodes (i.e. the pressure for an RFT) are internalized
        (irrespective of whether they are of type dynamic_state or
        dynamic_result).

   Observe that this cascade can result in some nodes, i.e. a rate we
   are observing, to be marked for internalization several times -
   that is no problem.
    
   -----
   
   For performance reason model_config contains two bool vectors
   __load_state and __load_result; if they are true the state and
   summary are loaded from disk, otherwise no loading is
   performed. This implies that if we do not want to internalize the
   full state but for instance the pressure (i.e. for an RFT) we must
   set the __load_state variable for the actual report step to
   true. For this reason calls enkf_config_node_internalize() must be
   accompanied by calls to model_config_set_load_state|results() -
   this is ensured when using this function to manipulate the
   configuration of internalization.

*/


void enkf_main_init_internalization( enkf_main_type * enkf_main , run_mode_type run_mode ) {
  /* Clearing old internalize flags. */
  model_config_init_internalization( enkf_main->model_config );
  ensemble_config_init_internalization( enkf_main->ensemble_config );
  
  /* Internalizing the initial state. */
  model_config_set_internalize_state( enkf_main->model_config , 0);
  
  
  /* We internalize all the endpoints in the enkf_sched. */
  {
    int inode;
    enkf_sched_type * enkf_sched = model_config_get_enkf_sched(enkf_main->model_config);
    for (inode = 0; inode < enkf_sched_get_num_nodes( enkf_sched ); inode++) {
      const enkf_sched_node_type * node = enkf_sched_iget_node(enkf_sched , inode);
      int report_step2            = enkf_sched_node_get_last_step( node );
      model_config_set_internalize_state( enkf_main->model_config , report_step2);
      model_config_set_internalize_results( enkf_main->model_config , report_step2);
    }
  }

  
  /* We internalize all the results - for all the report steps (beyond zero). */
  {
    int report_step;
    for (report_step = 1; report_step < enkf_main_get_total_length(enkf_main); report_step++)
      model_config_set_internalize_results( enkf_main->model_config , report_step);
  }

  
  /* Make sure we internalize at all observation times.*/
  {
    hash_type      * map  = enkf_obs_alloc_summary_map(enkf_main->obs);
    hash_iter_type * iter = hash_iter_alloc(map); 
    const char * obs_key = hash_iter_get_next_key(iter);
    
    while (obs_key != NULL) {
      obs_vector_type * obs_vector = enkf_obs_get_vector( enkf_main->obs , obs_key );
      enkf_config_node_type * data_node = obs_vector_get_config_node( obs_vector );
      int active_step = -1;
      do {
	active_step = obs_vector_get_next_active_step( obs_vector , active_step );
	if (active_step >= 0) {
	  enkf_config_node_set_internalize( data_node , active_step );
	  {
	    enkf_var_type var_type = enkf_config_node_get_var_type( data_node );
	    if (var_type == DYNAMIC_STATE)
	      model_config_set_load_state( enkf_main->model_config , active_step);
	    else if (var_type == DYNAMIC_RESULT)
	      model_config_set_load_results( enkf_main->model_config , active_step);
	  }
	}
      } while (active_step >= 0);
      obs_key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
    hash_free(map);
  }
}
  



  
