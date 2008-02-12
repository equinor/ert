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




struct enkf_main_struct {
  enkf_config_type   *config;
  meas_matrix_type   *meas_matrix;
  enkf_obs_type      *obs;
  obs_data_type      *obs_data;
  enkf_state_type   **state_list;
  sched_file_type    *sched_file;
  history_type       *hist;
  enkf_fs_type       *fs;

  thread_pool_type   *thread_pool_load_ecl;
  void_arg_type     **arg_load_ecl;
};


/*****************************************************************/






enkf_fs_type * enkf_main_get_fs_ref(const enkf_main_type * ens) { return ens->fs; }



enkf_main_type * enkf_main_alloc(enkf_config_type * config, enkf_fs_type *fs) {
  int ens_size               = enkf_config_get_ens_size(config);
  enkf_main_type * enkf_main = malloc(sizeof *enkf_main);
  enkf_main->config         = config;
  enkf_main->sched_file     = sched_file_alloc(enkf_config_get_start_date(config));
  sched_file_parse(enkf_main->sched_file , enkf_config_get_schedule_file(config));
  enkf_main->obs            = enkf_obs_fscanf_alloc(enkf_main->sched_file , enkf_config_get_obs_config_file(config));
  enkf_main->obs_data       = obs_data_alloc();
  enkf_main->fs             = fs;

  enkf_main->meas_matrix  = meas_matrix_alloc(ens_size);
  enkf_main->state_list   = malloc(ens_size * sizeof * enkf_main->state_list);
  {
    int iens;
    int iens_offset = enkf_config_get_ens_offset(config);
    for (iens = 0; iens < ens_size; iens++) {
      char * run_path 	    = enkf_config_alloc_run_path(config , iens + iens_offset);
      char * eclbase  	    = enkf_config_alloc_eclbase (config , iens + iens_offset);
      char * ecl_store_path = enkf_config_alloc_ecl_store_path (config , iens + iens_offset);
      enkf_main->state_list[iens] = enkf_state_alloc(config   , iens + iens_offset , enkf_config_iget_ecl_store(config , iens) , enkf_main->fs , 
						     run_path , 
						     eclbase  , 
						     ecl_store_path , 
						     meas_matrix_iget_vector(enkf_main->meas_matrix , iens));
      free(run_path);
      free(eclbase);
      free(ecl_store_path);
    }
  }
  enkf_main->thread_pool_load_ecl = NULL;
  enkf_main->arg_load_ecl         = NULL;
  return  enkf_main;

}







void enkf_main_add_data_kw(enkf_main_type * enkf_main , const char * new_kw , const char * value) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int iens;
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_add_data_kw(enkf_main->state_list[iens] , new_kw , value);
}


void enkf_main_set_data_kw(enkf_main_type * enkf_main , const char * new_kw , const char * value) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int iens;
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_set_data_kw(enkf_main->state_list[iens] , new_kw , value);
}


void enkf_main_init_eclipse(enkf_main_type * enkf_main) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  int iens;
  for (iens = 0; iens < ens_size; iens++)
    enkf_state_init_eclipse(enkf_main->state_list[iens]);
}




void enkf_main_free(enkf_main_type * enkf_main) {  
  enkf_obs_free(enkf_main->obs);
  {
    const int ens_size = enkf_config_get_ens_size(enkf_main->config);
    int i;
    for (i=0; i < ens_size; i++)
      enkf_state_free(enkf_main->state_list[i]);
    free(enkf_main->state_list);
  }
  sched_file_free(enkf_main->sched_file);
  meas_matrix_free(enkf_main->meas_matrix);
  obs_data_free(enkf_main->obs_data);
  free(enkf_main);
}



/*****************************************************************/

void enkf_main_load_ecl_init_mt(enkf_main_type * enkf_main , int report_step) {
  const int ens_size = enkf_config_get_ens_size(enkf_main->config);
  enkf_obs_get_observations(enkf_main->obs , report_step , enkf_main->obs_data);
  if (enkf_main->arg_load_ecl != NULL) {
    fprintf(stderr,"%s: hmmm - something is rotten - aborting \n",__func__);
    abort();
  }
  {
    int iens;
    enkf_main->arg_load_ecl = util_malloc(ens_size * sizeof * enkf_main->arg_load_ecl , __func__);
    for (iens = 0; iens < ens_size; iens++) {
      enkf_main->arg_load_ecl[iens] = void_arg_alloc4(pointer_value , pointer_value , int_value , bool_value);
      void_arg_pack_ptr(enkf_main->arg_load_ecl[iens]  , 0 , enkf_main->state_list[iens]);
      void_arg_pack_ptr(enkf_main->arg_load_ecl[iens]  , 1 , enkf_main->obs);
      void_arg_pack_int(enkf_main->arg_load_ecl[iens]  , 2 , report_step);
      void_arg_pack_bool(enkf_main->arg_load_ecl[iens] , 3 , enkf_config_get_unified(enkf_main->config));
    }
  }

  if (enkf_main->thread_pool_load_ecl != NULL) {
    fprintf(stderr,"%s: hmmm - something is rotten - aborting \n",__func__);
    abort();
  }
  enkf_main->thread_pool_load_ecl = thread_pool_alloc(ens_size);
}



void enkf_main_iload_ecl_mt(enkf_main_type *enkf_main , int iens) {
  thread_pool_add_job(enkf_main->thread_pool_load_ecl , enkf_state_load_ecl_void , enkf_main->arg_load_ecl[iens]);
}


void enkf_main_load_ecl_complete_mt(enkf_main_type *enkf_main) {
  thread_pool_join(enkf_main->thread_pool_load_ecl);
  thread_pool_free(enkf_main->thread_pool_load_ecl);
  enkf_main->thread_pool_load_ecl = NULL;

  {
    const int ens_size = enkf_config_get_ens_size(enkf_main->config);
    int iens;
    for (iens = 0; iens < ens_size; iens++)
      void_arg_free(enkf_main->arg_load_ecl[iens]);
    free(enkf_main->arg_load_ecl);
    enkf_main->arg_load_ecl = NULL;
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
