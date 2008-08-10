#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <util.h>
#include <hash.h>
#include <multz_config.h>
#include <enkf_main.h>
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
#include <pgbox_config.h>
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


struct enkf_main_struct {
  enkf_config_type   *config;
  job_queue_type     *job_queue;
  
  enkf_obs_type      *obs;
  meas_matrix_type   *meas_matrix;
  obs_data_type      *obs_data;
  enkf_state_type   **ensemble;
  sched_file_type    *sched_file;
  history_type       *hist;
  enkf_fs_type       *fs;

  thread_pool_type   *thread_pool;
};


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
             




enkf_main_type * enkf_main_alloc(enkf_config_type * config, enkf_fs_type *fs , job_queue_type * job_queue , ext_joblist_type * joblist) {
  int ens_size               = enkf_config_get_ens_size(config);
  enkf_main_type * enkf_main = malloc(sizeof *enkf_main);
  enkf_main->config         = config;
  enkf_main->sched_file     = sched_file_alloc(enkf_config_get_start_date(config));

  sched_file_parse(enkf_main->sched_file , enkf_config_get_schedule_src_file(config));

  enkf_main->hist           = history_alloc_from_schedule(enkf_main->sched_file);
  enkf_main->obs            = enkf_obs_fscanf_alloc(enkf_main->config , enkf_main->sched_file , enkf_main->hist);
  enkf_main->obs_data       = obs_data_alloc();
  enkf_main->fs             = fs;
  enkf_main->job_queue      = job_queue;
  enkf_main->meas_matrix    = meas_matrix_alloc(ens_size);
  enkf_main->ensemble       = malloc(ens_size * sizeof * enkf_main->ensemble);
  {
    int iens , keys , ik;
    int iens_offset = enkf_config_get_ens_offset(config);
    char **keylist  = enkf_config_alloc_keylist(config , &keys);
    msg_type * msg  = msg_alloc("Initializing member: ");
    msg_show(msg);
    for (iens = 0; iens < ens_size; iens++) {
      char * run_path       = enkf_config_alloc_run_path(config , iens + iens_offset);
      char * eclbase        = enkf_config_alloc_eclbase (config , iens + iens_offset);
      char * ecl_store_path = enkf_config_alloc_ecl_store_path (config , iens + iens_offset);
      msg_update_int(msg , "%03d" , iens);
      enkf_main->ensemble[iens] = enkf_state_alloc(config   , iens + iens_offset , enkf_config_iget_ecl_store(config , iens) , enkf_main->fs , 
               joblist  , 
               run_path , 
               eclbase  , 
               ecl_store_path , 
               meas_matrix_iget_vector(enkf_main->meas_matrix , iens));
      
      
      free(run_path);
      free(eclbase);
      free(ecl_store_path);
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
  
  
  {
    const char * init_file   = enkf_config_get_init_file(config);
    if (init_file == NULL) 
      util_abort("%s: INIT_FILE is not set - must either use INIT_FILE in config_file or EQUIL keyword.",__func__);
    {
      char * tmp_include     = util_alloc_joined_string((const char *[4]) {"  " , "'" , init_file , "' /"} , 4 , "");
      char * DATA_initialize = util_alloc_multiline_string((const char *[2]) {"INCLUDE" , tmp_include} , 2);

      enkf_main_set_data_kw(enkf_main , "INIT" , DATA_initialize);
      
      free(DATA_initialize);
      free(tmp_include);
    }
  }
  enkf_main_insert_data_kw(enkf_main , ens_size);
  enkf_main->thread_pool = NULL;
  return  enkf_main;

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


void enkf_main_swapout_ensemble(enkf_main_type * enkf_main , int mask) {
  int iens;
  for (iens = 0; iens < enkf_config_get_ens_size(enkf_main->config); iens++) 
    enkf_state_swapout(enkf_main->ensemble[iens] , mask);
}


void enkf_main_swapin_ensemble(enkf_main_type * enkf_main , int mask) {
  int iens;
  for (iens = 0; iens < enkf_config_get_ens_size(enkf_main->config); iens++) 
    enkf_state_swapin(enkf_main->ensemble[iens] , mask );
}



void enkf_main_fwrite_ensemble(enkf_main_type * enkf_main , int mask , int report_step , state_enum state) {
  int iens;
  for (iens = 0; iens < enkf_config_get_ens_size(enkf_main->config); iens++) 
    enkf_state_fwrite_as(enkf_main->ensemble[iens] , mask , report_step , state);
}


void  enkf_main_initialize_ensemble(enkf_main_type * enkf_main) {
  int iens;
  for (iens = 0; iens < enkf_config_get_ens_size(enkf_main->config); iens++) 
    enkf_state_initialize(enkf_main->ensemble[iens]);
  enkf_main_fprintf_results(enkf_main);
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

void enkf_main_fprintf_results(const enkf_main_type * enkf_main) {
  const int ens_size     = enkf_config_get_ens_size(enkf_main->config);
  const int report_step  = enkf_state_get_report_step(enkf_main->ensemble[0]);
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



/*****************************************************************/

void enkf_main_set_ensemble_state(enkf_main_type * enkf_main , int report_step , state_enum state) {
  int iens;
  for (iens = 0; iens < enkf_config_get_ens_size(enkf_main->config); iens++) 
    enkf_state_set_state(enkf_main->ensemble[iens] , report_step , state);
}

void enkf_main_update_ensemble(enkf_main_type * enkf_main , int step1 , int step2) {
  enkf_main_set_ensemble_state(enkf_main , step2 , forecast);
  enkf_main_swapin_ensemble(enkf_main , ecl_restart + ecl_summary + parameter);
}


/******************************************************************/

void enkf_main_run(enkf_main_type * enkf_main, int init_step , state_enum init_state , int step1 , int step2 , bool enkf_update, bool unlink_run_path , const stringlist_type * forward_model) {
  bool  load_results            = true; /** Must have individual switch */
  const int ens_size            = enkf_config_get_ens_size(enkf_main->config);
  enkf_run_info_type ** run_info;     
  int iens;
  
  printf("Starting forward step: %d -> %d \n",step1,step2);
  enkf_obs_get_observations(enkf_main->obs , step2 , enkf_main->obs_data);
  meas_matrix_reset(enkf_main->meas_matrix);

  run_info = util_malloc(ens_size * sizeof * run_info , __func__);
  if (enkf_main->thread_pool != NULL) 
    util_abort("%s: hmmm - something is rotten - aborting \n",__func__);
  
  for (iens = 0; iens < ens_size; iens++) 
    run_info[iens] = enkf_run_info_alloc(enkf_main->ensemble[iens] , enkf_main->job_queue , enkf_main->obs , enkf_main->sched_file , enkf_config_get_unified(enkf_main->config),
					 init_step , init_state , step1 , step2 , 
					 load_results , unlink_run_path , (stringlist_type *) forward_model);
  /*
    The thread pool can just be a local variable.
    No reason to bind it to the enkf_main object.
  */

  {
    {
      pthread_t queue_thread;
      void_arg_type * queue_args = void_arg_alloc2(void_pointer , int_value);
      void_arg_pack_ptr(queue_args , 0 , enkf_main->job_queue);
      void_arg_pack_int(queue_args , 1 , ens_size);
      
      pthread_create( &queue_thread , NULL , job_queue_run_jobs__ , queue_args);

      enkf_main->thread_pool = thread_pool_alloc(4);

      for (iens = 0; iens < ens_size; iens++) 
      {
        thread_pool_add_job(enkf_main->thread_pool , enkf_state_start_eclipse__ , run_info[iens]);
      }

      thread_pool_join(enkf_main->thread_pool);  /* OK: All directories for ECLIPSE simulations are ready. */
      thread_pool_free(enkf_main->thread_pool);

      enkf_main->thread_pool = thread_pool_alloc(ens_size);
      for (iens = 0; iens < ens_size; iens++) 
      {  
        thread_pool_add_job(enkf_main->thread_pool , enkf_state_complete_eclipse__ , run_info[iens]);
      }

      thread_pool_join(enkf_main->thread_pool);  /* All jobs have completed and the results have been loaded back. */
      pthread_join ( queue_thread , NULL );      /* The thread running the queue is complete.                      */
      job_queue_finalize(enkf_main->job_queue);  /* Must *NOT* be called before all jobs are done.                 */               
      void_arg_free( queue_args );
    }
    
    {
      bool complete_OK = true;
      int model_nr = 0;
      for (iens = 0; iens < ens_size; iens++) {
	if (! enkf_run_info_OK(run_info[iens])) { 
          if ( !complete_OK ) {
            fprintf(stderr,"Some models failed to integrate from DATES %d -> %d:\n",step1 , step2);
            complete_OK = false;
          }
          fprintf(stderr,"  %02d: %s \n",model_nr , enkf_config_alloc_run_path(enkf_main->config , iens));
          model_nr++;
          }
      }
      if (!complete_OK)
      {
        util_exit("");
      }
    }
  }

  /** Opprydding */
  for (iens = 0; iens < ens_size; iens++) 
    enkf_run_info_free(run_info[iens]);
  free(run_info);
  thread_pool_free(enkf_main->thread_pool);
  enkf_main->thread_pool = NULL;
  
  if (load_results) {
    enkf_main_swapin_ensemble(enkf_main , ecl_restart + ecl_summary + parameter);
    enkf_main_set_ensemble_state(enkf_main , step2 , forecast);
  }

  printf("Starter paa oppdatering \n");
  if (enkf_update) {
    double *X = analysis_allocX(ens_size , obs_data_get_nrobs(enkf_main->obs_data) , enkf_main->meas_matrix , enkf_main->obs_data , false , true);
    
    if (X != NULL) {
      /* The second to last argument is the number of double
	 we ask for, to get the number of bytes you must multiply
	 by eight.

	 1024 * 1024 * 128 => 1GB of memory
      */
      enkf_ensemble_update(enkf_main->ensemble , ens_size , 1024*1024*256 /* 2GB */, X);   
      free(X);
    }
  }
    
  printf("---------------------------------\n");
  if (enkf_update) {
    enkf_main_fwrite_ensemble(enkf_main , parameter + ecl_restart + ecl_summary , step2 , analyzed);
    enkf_main_fprintf_results(enkf_main);
  }
  printf("%s: ferdig med step: %d \n" , __func__,step2);
}




const sched_file_type * enkf_main_get_sched_file(const enkf_main_type * enkf_main) { return enkf_main->sched_file; }
