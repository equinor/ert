/**
   See the overview documentation of the observation system in enkf_obs.c
*/
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <util.h>
#include <stdio.h>
#include <summary_obs.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <summary.h>
#include <active_list.h>


#define SUMMARY_OBS_TYPE_ID 66103


struct summary_obs_struct {
  int       __type_id; 
  char    * summary_key;    /** The observation, in summary.x syntax, e.g. GOPR:FIELD.    */

  double    value;          /** Observation value. */
  double    std;            /** Standard deviation of observation. */
};





/**
  This function allocates a summary_obs instance. The summary_key
  string should be of the format used by the summary.x program.
  E.g., WOPR:P4 would condition on WOPR in well P4.

  Observe that this format is currently *not* checked before the actual
  observation time.

  TODO
  Should check summary_key on alloc.
*/
summary_obs_type * summary_obs_alloc(  
  const char   * summary_key,
  double value ,
  double std)
{
  summary_obs_type * obs = util_malloc(sizeof * obs , __func__);
  
  obs->__type_id     = SUMMARY_OBS_TYPE_ID; 
  obs->summary_key   = util_alloc_string_copy(summary_key);
  obs->value         = value;
  obs->std           = std;
  
  return obs;
}


SAFE_CAST(summary_obs   , SUMMARY_OBS_TYPE_ID);
IS_INSTANCE(summary_obs , SUMMARY_OBS_TYPE_ID);


void summary_obs_free(summary_obs_type * summary_obs) {
  free(summary_obs->summary_key);
  free(summary_obs);
}







const char * summary_obs_get_summary_key(const summary_obs_type * summary_obs)
{
  return summary_obs->summary_key;
}



void summary_obs_get_observations(const summary_obs_type * summary_obs,
				  int                      restart_nr,
				  obs_data_type          * obs_data,
				  const active_list_type * active_list) {
  char * obs_key = util_alloc_sprintf("%s:%d" , summary_obs->summary_key , restart_nr);
  obs_data_add(obs_data , summary_obs->value , summary_obs->std , obs_key);
  free( obs_key );
}



void summary_obs_measure(const summary_obs_type * obs,
			 const summary_type     * summary,
			 meas_vector_type       * meas_vector)
{
  meas_vector_add(meas_vector , summary_get(summary));
}


double summary_obs_chi2(const summary_obs_type * obs,
			const summary_type     * summary) {
  double x = (summary_get(summary) - obs->value) / obs->std;
  return x*x;
}






void summary_obs_user_get(const summary_obs_type * summary_obs , const char * index_key , double * value , double * std, bool * valid) {
  *valid = true;
  *value = summary_obs->value;
  *std   = summary_obs->std;
}



/*****************************************************************/

VOID_FREE(summary_obs)
VOID_GET_OBS(summary_obs)
VOID_USER_GET_OBS(summary_obs)
VOID_MEASURE(summary_obs , summary)
VOID_CHI2(summary_obs , summary)
