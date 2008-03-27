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
  meas_matrix_type   *meas_matrix;
  enkf_obs_type      *obs;
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
  sched_file_parse(enkf_main->sched_file , enkf_config_get_schedule_file(config));
  enkf_main->hist           = history_alloc_from_schedule(enkf_main->sched_file);
  enkf_main->obs            = enkf_obs_fscanf_alloc(enkf_main->config , enkf_main->sched_file , enkf_main->hist);
  enkf_main->obs_data       = obs_data_alloc();
  enkf_main->fs             = fs;
  enkf_main->ecl_queue      = ecl_queue;

  enkf_main->meas_matrix  = meas_matrix_alloc(ens_size);
  enkf_main->ensemble     = malloc(ens_size * sizeof * enkf_main->ensemble);
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
  enkf_main_add_data_kw(enkf_main , "INIT" , "INCLUDE\n   \'EQUIL.INC\'/\nRPTSOL\n     0  0  0  0  0 0   2  0   0  0   0   0  0   0 /\n");
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


/*****************************************************************/

void enkf_main_run(enkf_main_type * enkf_main, int step1 , int step2) {
  const int ens_size            = enkf_config_get_ens_size(enkf_main->config);
  const int sleep_time     	= 1;
  int iens;
  
  enkf_obs_get_observations(enkf_main->obs , step2 , enkf_main->obs_data);
  if (enkf_main->void_arg != NULL) {
    fprintf(stderr,"%s: hmmm - something is rotten - aborting \n",__func__);
    abort();
  }
  
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
  
  if (enkf_main->thread_pool != NULL) {
    fprintf(stderr,"%s: hmmm - something is rotten - aborting \n",__func__);
    abort();
  }
  enkf_main->thread_pool = thread_pool_alloc(ens_size);
  for (iens = 0; iens < ens_size; iens++) 
    thread_pool_add_job(enkf_main->thread_pool , enkf_state_run_eclipse__ , enkf_main->void_arg[iens]);

  ecl_queue_run_jobs(enkf_main->ecl_queue , ens_size);
  thread_pool_join(enkf_main->thread_pool);
  

  for (iens = 0; iens < ens_size; iens++) 
    void_arg_free(enkf_main->void_arg[iens]);
  free(enkf_main->void_arg);
  enkf_main->void_arg = NULL;
  thread_pool_free(enkf_main->thread_pool);
  enkf_main->thread_pool = NULL;
}

