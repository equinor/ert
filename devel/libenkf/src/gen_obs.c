#include <stdlib.h>
#include <enkf_util.h>
#include <enkf_types.h>
#include <enkf_macros.h>
#include <util.h>
#include <gen_data_config.h>
#include <gen_data.h>
#include <gen_obs.h>
#include <meas_vector.h>
#include <obs_data.h>
#include <gen_obs.h>
#include <gen_common.h>




struct gen_obs_struct {
  gen_data_config_type * config;
  /* 
     The data and std of the observations are just loaded in the
     get_obs function, used and discareded. Observe that the
     gen_data_config instance contains meta information which is
     related to the observation:

        __obs_size: This field is stored size of the last observation
	            vector, when data come in from the forward run, it
	            is checked that the size matches.

        iactive : A boolean vector of which elements in the
                  observation vector which are active/inactive.
  */
};



void gen_obs_free(gen_obs_type * gen_obs) {
  free(gen_obs);
}



gen_obs_type * gen_obs_alloc(gen_data_config_type * config) {
  gen_obs_type * obs = util_malloc(sizeof * obs , __func__);
  
  obs->config  = config;

  return obs;
}


void gen_obs_measure(const gen_obs_type * gen_obs , const gen_data_type * gen_data , meas_vector_type * meas_vector) {
  const gen_data_config_type * config = gen_obs->config;
  const int size = gen_data_config_get_data_size(config);
  if (size > 0) {
    const bool *iactive = gen_data_config_get_iactive( config );
    int id;
  
    for (id = 0; id < size; id++) {
      if (iactive[id])
	meas_vector_add( meas_vector , gen_data_iget_double( gen_data , id ));
    }
  }
}




void gen_obs_get_observations(gen_obs_type * gen_obs , int report_step, obs_data_type * obs_data) {
  /* Checks if the observation is globally ON or OFF - the enkf_obs_node might be a bit messed up on this question. */ 
  if (gen_data_config_obs_on(gen_obs->config , report_step)) {  
    char *file_tag;
    ecl_type_enum      ecl_type;
    int size;
    char * std;
    char * data;
    {
      bool               fortran_endian_flip;  
      gen_data_file_type file_type;
      char *obs_file , *config_tag;
    
      gen_data_config_get_obs_file(gen_obs->config , report_step , &obs_file , &config_tag);
      gen_common_get_file_type(obs_file , &file_type , &fortran_endian_flip);
      {
	FILE * stream = util_fopen( obs_file , "r");
      
	gen_common_fload_header( file_type , stream , config_tag , &file_tag , &size , &ecl_type);
	data = util_malloc(size * ecl_util_get_sizeof_ctype(ecl_type) , __func__);
	std  = util_malloc(size * ecl_util_get_sizeof_ctype(ecl_type) , __func__);
	gen_common_fload_data( stream , obs_file , file_type , ecl_type , size , data);
	gen_common_fload_data( stream , obs_file , file_type , ecl_type , size , std);
	gen_data_config_fload_iactive( gen_obs->config , stream , obs_file , file_type ,size);
	gen_data_config_set_obs_size(gen_obs->config , size);
	fclose( stream );
      }
    }
  
    {
      const bool *iactive = gen_data_config_get_iactive( gen_obs->config );
      int id;
    
      for (id = 0; id < size; id++) {
	if (iactive[id]) {
	  char * kw = util_alloc_sprintf("%s/%d" , file_tag , id);
	  double d  = gen_common_iget_double(id , size , ecl_type , data) ;
	  double s  = gen_common_iget_double(id , size , ecl_type , std);
	  obs_data_add(obs_data ,d , s, kw);
	  free(kw);
	}
      }
    }
    free(file_tag);
    free(data);
    free(std);
  }
}
  
/*****************************************************************/
VOID_FREE(gen_obs)
VOID_GET_OBS(gen_obs)
VOID_MEASURE(gen_obs , gen_data)
