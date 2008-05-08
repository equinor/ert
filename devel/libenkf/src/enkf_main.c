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
#include <ecl_queue.h>

struct enkf_main_struct {
  enkf_config_type   *config;
  ecl_queue_type     *ecl_queue;
  enkf_obs_type      *obs;
  meas_matrix_type   *meas_matrix;
  obs_data_type      *obs_data;
  enkf_state_type   **ensemble;
  sched_file_type    *sched_file;
  history_type       *hist;
  enkf_fs_type       *fs;

  thread_pool_type   *thread_pool;
  void_arg_type     **void_arg;
};


/*****************************************************************/






enkf_fs_type * enkf_main_get_fs_ref(const enkf_main_type * ens) { return ens->fs; }



enkf_main_type * enkf_main_alloc(enkf_config_type * config, enkf_fs_type *fs , ecl_queue_type * ecl_queue) {
  int ens_size               = enkf_config_get_ens_size(config);
  enkf_main_type * enkf_main = malloc(sizeof *enkf_main);
  enkf_main->config         = config;
  enkf_main->sched_file     = sched_file_alloc(enkf_config_get_start_date(config));
  sched_file_parse(enkf_main->sched_file , enkf_config_get_schedule_src_file(config));
  enkf_main->hist           = history_alloc_from_schedule(enkf_main->sched_file);
  enkf_main->obs            = enkf_obs_fscanf_alloc(enkf_main->config , enkf_main->sched_file , enkf_main->hist);
  enkf_main->obs_data       = obs_data_alloc();
  enkf_main->fs             = fs;
  enkf_main->ecl_queue      = ecl_queue;

  enkf_main->meas_matrix  = meas_matrix_alloc(ens_size);
  enkf_main->ensemble     = malloc(ens_size * sizeof * enkf_main->ensemble);
  enkf_config_post_check(config , enkf_main->sched_file);
  {
    int iens , keys , ik;
    int iens_offset = enkf_config_get_ens_offset(config);
    char **keylist = enkf_config_alloc_keylist(config , &keys);
    for (iens = 0; iens < ens_size; iens++) {
      char * run_path 	    = enkf_config_alloc_run_path(config , iens + iens_offset);
      char * eclbase  	    = enkf_config_alloc_eclbase (config , iens + iens_offset);
      char * ecl_store_path = enkf_config_alloc_ecl_store_path (config , iens + iens_offset);
      enkf_main->ensemble[iens] = enkf_state_alloc(config   , iens + iens_offset , enkf_config_iget_ecl_store(config , iens) , enkf_main->fs , 
						   run_path , 
						   eclbase  , 
						   ecl_store_path , 
						   meas_matrix_iget_vector(enkf_main->meas_matrix , iens));
      
      
      free(run_path);
      free(eclbase);
      free(ecl_store_path);
    }
    for (ik = 0; ik < keys; ik++) {
      const enkf_config_node_type * config_node = enkf_config_get_node_ref(config , keylist[ik]);
      for (iens = 0; iens < ens_size; iens++)
	enkf_state_add_node(enkf_main->ensemble[iens] , keylist[ik] , config_node);
    }
    
    util_free_string_list(keylist , keys);
  }
  
  
  /*
    This - can not be hardcoded ....
  */
  {
    char * DATA_initialize = util_alloc_multiline_string((const char *[3]) {"EQUIL" , 
									    "       2469   382.4   1705.0  0.0    500    0.0     1     1      20 /",
									    "       2469   382.4   1000.0  0.0    500    0.0     1     1      20 /"} , 3);      
    enkf_main_add_data_kw(enkf_main , "INIT" , DATA_initialize);
    free(DATA_initialize);
  }
  enkf_main_add_data_kw(enkf_main , "INCLUDE_PATH" , "/h/a152128/EnKF/devel/EnKF/libenkf/src/Gurbat");
  enkf_main->thread_pool = NULL;
  enkf_main->void_arg    = NULL;
  return  enkf_main;

}







void enkf_main_add_data_kw(enkf_main_type * enkf_main , const char * new_kw , const char * value) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int iens;
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_add_data_kw(enkf_main->ensemble[iens] , new_kw , value);
}


void enkf_main_set_data_kw(enkf_main_type * enkf_main , const char * new_kw , const char * value) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int iens;
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_set_data_kw(enkf_main->ensemble[iens] , new_kw , value);
}


void enkf_main_init_eclipse(enkf_main_type * enkf_main , int report_step1 , int report_step2) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int iens;
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_init_eclipse(enkf_main->ensemble[iens] , enkf_main->sched_file , report_step1 , report_step2);
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
  free(enkf_main);
}



/*****************************************************************/

void enkf_main_load_ecl_init_mt(enkf_main_type * enkf_main , int report_step) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  enkf_obs_get_observations(enkf_main->obs , report_step , enkf_main->obs_data);
  if (enkf_main->void_arg != NULL) {
    fprintf(stderr,"%s: hmmm - something is rotten - aborting \n",__func__);
    abort();
  }
  {
    int iens;
    enkf_main->void_arg = util_malloc(ens_size * sizeof * enkf_main->void_arg , __func__);
    for (iens = 0; iens < ens_size; iens++) {
      enkf_main->void_arg[iens] = void_arg_alloc4(void_pointer , void_pointer , int_value , bool_value);
      void_arg_pack_ptr(enkf_main->void_arg[iens]  , 0 , enkf_main->ensemble[iens]);
      void_arg_pack_ptr(enkf_main->void_arg[iens]  , 1 , enkf_main->obs);
      void_arg_pack_int(enkf_main->void_arg[iens]  , 2 , report_step);
      void_arg_pack_bool(enkf_main->void_arg[iens] , 3 , enkf_config_get_unified(enkf_main->config));
    }
  }

  if (enkf_main->thread_pool != NULL) {
    fprintf(stderr,"%s: hmmm - something is rotten - aborting \n",__func__);
    abort();
  }
  enkf_main->thread_pool = thread_pool_alloc(ens_size);
}



void enkf_main_iload_ecl_mt(enkf_main_type *enkf_main , int iens) {
  thread_pool_add_job(enkf_main->thread_pool , enkf_state_load_ecl_void , enkf_main->void_arg[iens]);
}


void enkf_main_load_ecl_complete_mt(enkf_main_type *enkf_main) {
  thread_pool_join(enkf_main->thread_pool);
  thread_pool_free(enkf_main->thread_pool);
  enkf_main->thread_pool = NULL;
  
  {
    const int ens_size = enkf_config_get_ens_size(enkf_main->config);
    int iens;
    for (iens = 0; iens < ens_size; iens++)
      void_arg_free(enkf_main->void_arg[iens]);
    free(enkf_main->void_arg);
    enkf_main->void_arg = NULL;
  }

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


void enkf_main_add_well_obs(enkf_main_type * enkf_main , const char *obs_key , const char * obs_label , const char * config_file) {
  const enkf_config_node_type * config_node = enkf_main_assert_obs(enkf_main , obs_key , WELL);
  enkf_obs_add_well_obs(enkf_main->obs , config_node , obs_key , obs_label , config_file);
}


void enkf_main_add_field_obs(enkf_main_type * enkf_main, const char * obs_key, const char * obs_label , int size, const int *i , const int *j , const int *k, const double * obs_data , time_t meas_time) { 
  const enkf_config_node_type * config_node = enkf_main_assert_obs(enkf_main , obs_key , FIELD);
  enkf_obs_add_field_obs(enkf_main->obs , config_node , obs_key , obs_label , size , i , j , k , obs_data , meas_time);
}


void enkf_main_add_rft_obs(enkf_main_type * enkf_main , const ecl_rft_node_type * rft_node, const double * p_data) {
  const enkf_config_node_type * config_node = enkf_main_assert_obs(enkf_main , "PRES" , FIELD);
  enkf_obs_add_rft_obs(enkf_main->obs , config_node , rft_node , p_data);
}




/*****************************************************************/

void enkf_main_analysis(enkf_main_type * enkf_main) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int nrobs          = obs_data_get_nrobs(enkf_main->obs_data);
  
  if (nrobs > 0) {
    double * X = analysis_allocX(ens_size , obs_data_get_nrobs(enkf_main->obs_data) , enkf_main->meas_matrix , enkf_main->obs_data , true , true);
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

/*****************************************************************/


void * dummy_sleep(void * arg) {
  sleep( 5 );
}


void dummy_test() {
#define N 1500
  pthread_attr_t thread_attr;
  pthread_t t[N];
  int threads;
  int i = 0;
  int arg; 

  arg = i;
  pthread_attr_init( &thread_attr );
  pthread_attr_setdetachstate( &thread_attr , PTHREAD_CREATE_DETACHED );
  pthread_attr_setdetachstate( &thread_attr , PTHREAD_CREATE_JOINABLE );
  while ( ( pthread_create( &t[i] , &thread_attr , dummy_sleep , &arg) == 0) ) {
    i++;
    arg = i;
  }
  threads = i;
  pthread_attr_destroy ( &thread_attr );


  printf("*****************************************************************\n");
  printf("** Have created:%d threads                                     **\n" , i);
  printf("*****************************************************************\n");
  sleep(10);
  printf("All jobs *SHOULD* be complete ...?=?? \n");
  /*
    for (i=0; i < threads; i++)
    pthread_join( t[i] , NULL);
  */

}

/******************************************************************/

void enkf_main_run(enkf_main_type * enkf_main, int step1 , int step2) {
  const int ens_size            = enkf_config_get_ens_size(enkf_main->config);
  const int sleep_time     	= 1;
  int iens;
  
  /*dummy_test();*/
  enkf_obs_get_observations(enkf_main->obs , step2 , enkf_main->obs_data);
  meas_matrix_reset(enkf_main->meas_matrix);
  
  if (enkf_main->void_arg != NULL) 
    util_abort("%s: hmmm - something is rotten - aborting \n",__func__);
  if (enkf_main->thread_pool != NULL) 
    util_abort("%s: hmmm - something is rotten - aborting \n",__func__);
  
  enkf_main->void_arg = util_malloc(ens_size * sizeof * enkf_main->void_arg , __func__);
  for (iens = 0; iens < ens_size; iens++) {
    enkf_main->void_arg[iens] = void_arg_alloc9(void_pointer , void_pointer , void_pointer , void_pointer , bool_value , int_value , int_value , int_value , bool_value);
    void_arg_pack_ptr(enkf_main->void_arg[iens]  , 0 , enkf_main->ensemble[iens]);
    void_arg_pack_ptr(enkf_main->void_arg[iens]  , 1 , enkf_main->ecl_queue);
    void_arg_pack_ptr(enkf_main->void_arg[iens]  , 2 , enkf_main->obs);
    void_arg_pack_ptr(enkf_main->void_arg[iens]  , 3 , enkf_main->sched_file);
    void_arg_pack_bool(enkf_main->void_arg[iens] , 4 , enkf_config_get_unified(enkf_main->config));
    void_arg_pack_int(enkf_main->void_arg[iens]  , 5 , step1);
    void_arg_pack_int(enkf_main->void_arg[iens]  , 6 , step2);
    void_arg_pack_int(enkf_main->void_arg[iens]  , 7 , -1);
    /* 
       The final bool is not packed - that is a return value, which
       is set in the called routine.
    */
  }
  
  {
    pthread_t queue_thread;
    void_arg_type * queue_args = void_arg_alloc2(void_pointer , int_value);
    void_arg_pack_ptr(queue_args , 0 , enkf_main->ecl_queue);
    void_arg_pack_int(queue_args , 1 , ens_size);

    pthread_create( &queue_thread , NULL , ecl_queue_run_jobs__ , queue_args);

    enkf_main->thread_pool = thread_pool_alloc(4);
    for (iens = 0; iens < ens_size; iens++) 
      thread_pool_add_job(enkf_main->thread_pool , enkf_state_start_eclipse__ , enkf_main->void_arg[iens]);

    thread_pool_join(enkf_main->thread_pool);
    thread_pool_free(enkf_main->thread_pool);
    
    enkf_main->thread_pool = thread_pool_alloc(ens_size);
    for (iens = 0; iens < ens_size; iens++) 
      thread_pool_add_job(enkf_main->thread_pool , enkf_state_complete_eclipse__ , enkf_main->void_arg[iens]);
    thread_pool_join(enkf_main->thread_pool);
    pthread_join ( queue_thread , NULL );
    ecl_queue_finalize(enkf_main->ecl_queue);  /* Must *NOT* be called before all jobs are done */

    {
      bool complete_OK = true;
      for (iens = 0; iens < ens_size; iens++)
	complete_OK = (complete_OK && void_arg_get_bool(enkf_main->void_arg[iens] , 8));
      
      if ( !complete_OK) {
	fprintf(stderr,"Some models failed to integrate from DATES %d -> %d \n",step1 , step2);
	fprintf(stderr,"look in the ECLIPSE .PRT files - FIX the problem - and try again ... :-) \n");
	exit(1);
      }
    }
  }

  /** Opprydding */
  for (iens = 0; iens < ens_size; iens++) 
    void_arg_free(enkf_main->void_arg[iens]);
  free(enkf_main->void_arg);
  enkf_main->void_arg = NULL;
  thread_pool_free(enkf_main->thread_pool);
  enkf_main->thread_pool = NULL;

  /*dummy_test();*/

  
  enkf_main_swapin_ensemble(enkf_main , ecl_restart + ecl_summary + parameter);
  enkf_main_set_ensemble_state(enkf_main , step2 , forecast);
  printf("OK - er klar til aa allokere X \n");
  {
    double *X = analysis_allocX(ens_size , obs_data_get_nrobs(enkf_main->obs_data) , enkf_main->meas_matrix , enkf_main->obs_data , false , true);
    /*
      double * X = util_malloc(ens_size * ens_size * sizeof *X , __func__);
    */
    
    
    if (X != NULL) {
      /*
	{
	int i;
	for (i=0; i < ens_size*ens_size; i++) X[i] = 0.0;
	for (i=0; i < ens_size; i++) X[i * (ens_size + 1)] = 1.0;
	}
      */

      enkf_ensemble_update(enkf_main->ensemble , ens_size , 1024*1024*1024 /* 1GB */ , X);
      free(X);
    }
    
  }


  
  printf("---------------------------------\n");
  enkf_main_fwrite_ensemble(enkf_main , parameter + ecl_restart + ecl_summary , step2 , analyzed);
  printf("%s: ferdig med step: %d \n" , __func__,step2);
  /*dummy_test();*/
}


