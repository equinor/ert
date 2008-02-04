#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <util.h>
#include <hash.h>
#include <multz_config.h>
#include <enkf_ensemble.h>
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




struct enkf_ensemble_struct {
  int  		      ens_size;
  int                 iens_offset;
  meas_matrix_type   *meas_matrix;
  hash_type          *config_hash;
  enkf_obs_type      *obs;
  obs_data_type      *obs_data;
  enkf_state_type   **state_list;
  sched_file_type    *sched_file;
  
  char            **well_list;
  int               Nwells;

  enkf_fs_type     *fs;
  path_fmt_type    *run_path;
  path_fmt_type    *eclbase;
  path_fmt_type    *index_path;
  bool              endian_swap;
  bool              fmt_file;
  bool              unified;
  char *            data_file;

  thread_pool_type  *thread_pool_load_ecl;
  void_arg_type    **arg_load_ecl;
};


/*****************************************************************/




enkf_impl_type enkf_ensemble_impl_type(const enkf_ensemble_type *enkf_ensemble, const char * ecl_kw_name) {
  enkf_impl_type impl_type;

  if (hash_has_key(enkf_ensemble->config_hash , ecl_kw_name)) {
    enkf_config_node_type * node = hash_get(enkf_ensemble->config_hash , ecl_kw_name);
    impl_type = enkf_config_node_get_impl_type(node);
  } else
    impl_type = STATIC;

  return impl_type;
}



/*****************************************************************/
static void enkf_ensemble_realloc_well_list(enkf_ensemble_type * enkf_ensemble) {
  enkf_ensemble->well_list = realloc(enkf_ensemble->well_list , enkf_ensemble->Nwells * sizeof * enkf_ensemble->well_list);
}

bool enkf_ensemble_get_endian_swap(const enkf_ensemble_type * enkf_ensemble) { return enkf_ensemble->endian_swap; }
bool enkf_ensemble_get_fmt_file(const enkf_ensemble_type * enkf_ensemble) { return enkf_ensemble->fmt_file; }


enkf_fs_type * enkf_ensemble_get_fs_ref(const enkf_ensemble_type * ens) { return ens->fs; }

const char * enkf_ensemble_get_data_file(const enkf_ensemble_type * ens) { return ens->data_file; }



void enkf_ensemble_set_state_eclbase(const enkf_ensemble_type * ens , int iens) {
  char * eclbase = path_fmt_alloc_path(ens->eclbase , iens + ens->iens_offset);
  enkf_state_set_eclbase(ens->state_list[iens] , eclbase); 
  free(eclbase);
}


enkf_ensemble_type * enkf_ensemble_alloc(int ens_size , enkf_fs_type *fs, 
					 const char * data_file  , 
					 const char * _run_path   , 
					 const char * _eclbase    , 
					 sched_file_type * sched_file , 
					 bool fmt_file ,
					 bool unified  , 
					 bool endian_swap) {

  enkf_ensemble_type * enkf_ensemble = malloc(sizeof *enkf_ensemble);
  enkf_ensemble->config_hash    = hash_alloc(10);
  enkf_ensemble->sched_file     = sched_file; 
  enkf_ensemble->obs            = enkf_obs_alloc(enkf_ensemble->sched_file);
  enkf_ensemble->obs_data       = obs_data_alloc();
  enkf_ensemble->fs             = fs;

  enkf_ensemble->ens_size      = ens_size;
  enkf_ensemble->endian_swap   = endian_swap;
  enkf_ensemble->Nwells        = 0;
  enkf_ensemble->well_list     = NULL;
  enkf_ensemble_realloc_well_list(enkf_ensemble);
  enkf_ensemble->unified       = unified;
  enkf_ensemble->fmt_file      = fmt_file;
  enkf_ensemble->data_file     = util_alloc_string_copy(data_file);
  
  enkf_ensemble->run_path     = path_fmt_alloc_directory_fmt(_run_path , true);
  enkf_ensemble->eclbase      = path_fmt_alloc_file_fmt(_eclbase);
  enkf_ensemble->meas_matrix  = meas_matrix_alloc(enkf_ensemble->ens_size);
  enkf_ensemble->state_list   = malloc(enkf_ensemble->ens_size * sizeof * enkf_ensemble->state_list);
  enkf_ensemble->iens_offset  = 91;
  {
    int iens;
    for (iens = 0; iens < enkf_ensemble->ens_size; iens++) {
      char * run_path   = path_fmt_alloc_path(enkf_ensemble->run_path   , iens + enkf_ensemble->iens_offset);
      char * eclbase    = path_fmt_alloc_path(enkf_ensemble->eclbase    , iens + enkf_ensemble->iens_offset);

      enkf_ensemble->state_list[iens] = enkf_state_alloc(enkf_ensemble , iens + enkf_ensemble->iens_offset , enkf_ensemble->fs , 
							 run_path , eclbase ,
							 meas_matrix_iget_vector(enkf_ensemble->meas_matrix , iens));
      
      free(run_path);
      free(eclbase);
    }
  }
  enkf_ensemble->thread_pool_load_ecl = NULL;
  enkf_ensemble->arg_load_ecl         = NULL;
  return  enkf_ensemble;

}



bool enkf_ensemble_has_key(const enkf_ensemble_type * enkf_ensemble , const char * key) {
  return hash_has_key(enkf_ensemble->config_hash , key);
}


const char ** enkf_ensemble_get_well_list_ref(const enkf_ensemble_type * ens , int *Nwells) {
  *Nwells = ens->Nwells;
  return (const char **) ens->well_list;
}


void enkf_ensemble_add_well(enkf_ensemble_type * enkf_ensemble , const char *well_name , int size, const char ** var_list) {
  enkf_ensemble_add_type(enkf_ensemble , well_name , ecl_summary , WELL, NULL , well_config_alloc(well_name , size , var_list));
  enkf_ensemble->Nwells++;
  enkf_ensemble_realloc_well_list(enkf_ensemble);
  enkf_ensemble->well_list[enkf_ensemble->Nwells - 1] = util_alloc_string_copy(well_name);
}


void enkf_ensemble_add_gen_kw(enkf_ensemble_type * enkf_ensemble , const char * config_file) {
  enkf_ensemble_add_type(enkf_ensemble , "GEN_KW" , parameter , GEN_KW , NULL , gen_kw_config_fscanf_alloc(config_file , NULL));
}


void enkf_ensemble_add_data_kw(enkf_ensemble_type * enkf_ensemble , const char * new_kw , const char * value) {
  int iens;
  for (iens = 0; iens < enkf_ensemble->ens_size; iens++)
    enkf_state_add_data_kw(enkf_ensemble->state_list[iens] , new_kw , value);
}


void enkf_ensemble_set_data_kw(enkf_ensemble_type * enkf_ensemble , const char * new_kw , const char * value) {
  int iens;
  for (iens = 0; iens < enkf_ensemble->ens_size; iens++)
    enkf_state_set_data_kw(enkf_ensemble->state_list[iens] , new_kw , value);
}


void enkf_ensemble_init_eclipse(enkf_ensemble_type * enkf_ensemble) {
  int iens;
  for (iens = 0; iens < enkf_ensemble->ens_size; iens++)
    enkf_state_init_eclipse(enkf_ensemble->state_list[iens]);
}


void enkf_ensemble_add_type(enkf_ensemble_type * enkf_ensemble , 
		       const char    * key      , 
		       enkf_var_type enkf_type  , 
		       enkf_impl_type impl_type , 
		       const char   * ecl_file  , 
		       const void   * data) {
  if (enkf_ensemble_has_key(enkf_ensemble , key)) {
    fprintf(stderr,"%s: a ensuration object:%s has already been added - aborting \n",__func__ , key);
    abort();
  }

  {
    config_free_ftype * freef;
    switch(impl_type) {
    case(FIELD):
      freef             = field_config_free__;
      break;
    case(MULTZ):
      freef             = multz_config_free__;
      break;
    case(WELL):
      freef             = well_config_free__;
      break;
    case(MULTFLT):
      freef             = multflt_config_free__;
      break;
    case(EQUIL):
      freef             = equil_config_free__;
      break;
    case(STATIC):
      freef             = ecl_static_kw_config_free__;
      break;
    case(PGBOX):
      freef             = pgbox_config_free__;
      break;
    case(GEN_KW):
      freef             = gen_kw_config_free__;
      break;
    default:
      fprintf(stderr,"%s : invalid implementation type: %d - aborting \n",__func__ , impl_type);
      abort();
    }
    {
      enkf_config_node_type * node = enkf_config_node_alloc(enkf_type , impl_type , key , ecl_file , data , freef);
      hash_insert_hash_owned_ref(enkf_ensemble->config_hash , key , node , enkf_config_node_free__);
    }
  }
}



void enkf_ensemble_add_type0(enkf_ensemble_type * enkf_ensemble , const char *key , int size, enkf_var_type enkf_type , enkf_impl_type impl_type) {
  switch(impl_type) {
  case(STATIC):
    enkf_ensemble_add_type(enkf_ensemble , key , enkf_type , impl_type , NULL , ecl_static_kw_config_alloc(size , key , key));
    break;
  case(FIELD):
    fprintf(stderr,"%s: Can not add FIELD ens objects like:%s on the run - these must be from the main program with enkf_ensemble_add_type - sorry.\n",__func__ , key);
    abort();
    break;
  default:
    fprintf(stderr,"%s only STATIC and FIELD types are implemented - aborting \n",__func__);
    abort();
  }
}



void enkf_ensemble_free(enkf_ensemble_type * enkf_ensemble) {  
  hash_free(enkf_ensemble->config_hash);
  enkf_obs_free(enkf_ensemble->obs);
  {
    int i;
    for (i=0; i < enkf_ensemble->Nwells; i++)
      free(enkf_ensemble->well_list[i]);
    free(enkf_ensemble->well_list);
    
    for (i=0; i < enkf_ensemble->ens_size; i++)
      enkf_state_free(enkf_ensemble->state_list[i]);
    free(enkf_ensemble->state_list);
  }
  meas_matrix_free(enkf_ensemble->meas_matrix);
  path_fmt_free(enkf_ensemble->run_path);
  path_fmt_free(enkf_ensemble->eclbase);
  obs_data_free(enkf_ensemble->obs_data);
  free(enkf_ensemble->data_file);
  free(enkf_ensemble);
}



const enkf_config_node_type * enkf_ensemble_get_config_ref(const enkf_ensemble_type * ens, const char * key) {
  if (hash_has_key(ens->config_hash , key)) {
    enkf_config_node_type * node = hash_get(ens->config_hash , key);
    return node;
  } else {
    fprintf(stderr,"%s: ens node:%s does not exist \n",__func__ , key);
    abort();
  }
}

/*****************************************************************/

void enkf_ensemble_load_ecl_init_mt(enkf_ensemble_type * enkf_ensemble , int report_step) {
  enkf_obs_get_observations(enkf_ensemble->obs , report_step , enkf_ensemble->obs_data);
  if (enkf_ensemble->arg_load_ecl != NULL) {
    fprintf(stderr,"%s: hmmm - something is rotten - aborting \n",__func__);
    abort();
  }
  {
    int iens;
    enkf_ensemble->arg_load_ecl = util_malloc(enkf_ensemble->ens_size * sizeof * enkf_ensemble->arg_load_ecl , __func__);
    for (iens = 0; iens < enkf_ensemble->ens_size; iens++) {
      enkf_ensemble->arg_load_ecl[iens] = void_arg_alloc4(pointer_value , pointer_value , int_value , bool_value);
      void_arg_pack_ptr(enkf_ensemble->arg_load_ecl[iens]  , 0 , enkf_ensemble->state_list[iens]);
      void_arg_pack_ptr(enkf_ensemble->arg_load_ecl[iens]  , 1 , enkf_ensemble->obs);
      void_arg_pack_int(enkf_ensemble->arg_load_ecl[iens]  , 2 , report_step);
      void_arg_pack_bool(enkf_ensemble->arg_load_ecl[iens] , 3 , enkf_ensemble->unified);
    }
  }

  if (enkf_ensemble->thread_pool_load_ecl != NULL) {
    fprintf(stderr,"%s: hmmm - something is rotten - aborting \n",__func__);
    abort();
  }
  enkf_ensemble->thread_pool_load_ecl = thread_pool_alloc(enkf_ensemble->ens_size);
}



void enkf_ensemble_iload_ecl_mt(enkf_ensemble_type *enkf_ensemble , int iens) {
  thread_pool_add_job(enkf_ensemble->thread_pool_load_ecl , enkf_state_load_ecl_void , enkf_ensemble->arg_load_ecl[iens]);
}


void enkf_ensemble_load_ecl_complete_mt(enkf_ensemble_type *enkf_ensemble) {
  thread_pool_join(enkf_ensemble->thread_pool_load_ecl);
  thread_pool_free(enkf_ensemble->thread_pool_load_ecl);
  enkf_ensemble->thread_pool_load_ecl = NULL;

  {
    int iens;
    for (iens = 0; iens < enkf_ensemble->ens_size; iens++)
      void_arg_free(enkf_ensemble->arg_load_ecl[iens]);
    free(enkf_ensemble->arg_load_ecl);
    enkf_ensemble->arg_load_ecl = NULL;
  }

}



/*****************************************************************/


static const enkf_config_node_type * enkf_ensemble_assert_obs(const enkf_ensemble_type * ens , const char *obs_key , enkf_impl_type impl_type) {
  if (enkf_ensemble_has_key(ens , obs_key)) {
    const enkf_config_node_type * config_node = enkf_ensemble_get_config_ref(ens , obs_key);
    if (enkf_config_node_get_impl_type(config_node) == impl_type) 
      return config_node;
    else {
      fprintf(stderr,"%s ensemble object:%s exists - but it is not of correct type - aborting \n",__func__ , obs_key);
      abort();
    }
  } else {
    fprintf(stderr,"%s: ensemble does not have key: %s - aborting \n",__func__ , obs_key);
    abort();
  }
}


void enkf_ensemble_add_well_obs(enkf_ensemble_type * ens , const char *obs_key , const char * obs_label , const char * config_file) {
  const enkf_config_node_type * config_node = enkf_ensemble_assert_obs(ens , obs_key , WELL);
  enkf_obs_add_well_obs(ens->obs , config_node , obs_key , obs_label , config_file);
}


void enkf_ensemble_add_field_obs(enkf_ensemble_type * ens, const char * obs_key, const char * obs_label , int size, const int *i , const int *j , const int *k, const double * obs_data , time_t meas_time) { 
  const enkf_config_node_type * config_node = enkf_ensemble_assert_obs(ens , obs_key , FIELD);
  enkf_obs_add_field_obs(ens->obs , config_node , obs_key , obs_label , size , i , j , k , obs_data , meas_time);
}


void enkf_ensemble_add_rft_obs(enkf_ensemble_type * ens , const ecl_rft_node_type * rft_node, const double * p_data) {
  const enkf_config_node_type * config_node = enkf_ensemble_assert_obs(ens , "PRES" , FIELD);
  enkf_obs_add_rft_obs(ens->obs , config_node , rft_node , p_data);
}




/*****************************************************************/

void enkf_ensemble_analysis(enkf_ensemble_type * ens) {
  int nrobs = obs_data_get_nrobs(ens->obs_data);
  if (nrobs > 0) {
    double * X = analysis_allocX(ens->ens_size , obs_data_get_nrobs(ens->obs_data) , ens->meas_matrix , ens->obs_data , true , true);
    
    free(X);
  }
}
