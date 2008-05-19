#include <stdlib.h>
#include <util.h>
#include <stdio.h>
#include <summary_obs.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <summary.h>



struct summary_obs_struct {
  const summary_config_type * config;
  char   		    * var_string;
  double 		      obs_value;
  double 		      obs_std;         
};




/**
  This function allocates a summary_obs instance. The var string
  should be of the format WOPR:OP_4 used by the summary.x
  program. Observe that this format is *not* checked before the actual
  observation time.
*/


summary_obs_type * summary_obs_alloc(const summary_config_type * config , const char * var_string , double obs_value , double obs_std) {
  summary_obs_type * obs = util_malloc(sizeof * obs , __func__);
  
  obs->config     = config;
  obs->var_string = util_alloc_string_copy(var_string);
  obs->obs_value  = obs_value;
  obs->obs_std    = obs_std;

  return obs;
}



void summary_obs_free(summary_obs_type * obs) {
  free(obs->var_string);
  free(obs);
}



void summary_obs_get_observations(const summary_obs_type * summary_obs , int report_step, obs_data_type * obs_data) {
  obs_data_add(obs_data , summary_obs->obs_value , summary_obs->obs_std , summary_obs->var_string);
}



void summary_obs_measure(const summary_obs_type * obs , const summary_type * summary , meas_vector_type * meas_vector) {
  meas_vector_add(meas_vector , summary_get(summary , obs->var_string));
}


/**
  An ugly hack because the summary_type object is not defined ...
*/

VOID_FREE(summary_obs)
VOID_GET_OBS(summary_obs)
VOID_MEASURE(summary)
