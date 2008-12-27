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



void summary_obs_get_observations(
  const summary_obs_type * summary_obs,
  int                      restart_nr,
  obs_data_type          * obs_data)
{
  obs_data_add(obs_data , summary_obs->value , summary_obs->std , summary_obs->summary_key);
}



void summary_obs_measure(
  const summary_obs_type * obs,
  const summary_type     * summary,
  meas_vector_type       * meas_vector)
{
  meas_vector_add(meas_vector , summary_get(summary));
}



//summary_obs_type * summary_obs_alloc_from_HISTORY_OBSERVATION(
//  const conf_instance_type * conf_instance,
//  const history_type       * history)
//{
//  if(!conf_instance_is_of_class(conf_instance, "HISTORY_OBSERVATION"))
//    util_abort("%s: internal error. expected \"HISTORY_OBSERVATION\" instance, got \"%s\".\n",
//               __func__, conf_instance_get_class_name_ref(conf_instance) );
//
//
//  int          size;
//  double     * value;
//  double     * std;
//  bool       * default_used;
//
//  double         error      = conf_instance_get_item_value_double(conf_instance, "ERROR"     );
//  double         error_min  = conf_instance_get_item_value_double(conf_instance, "ERROR_MIN" );
//  const char * __error_mode = conf_instance_get_item_value_ref(   conf_instance, "ERROR_MODE");
//  const char *   sum_key    = conf_instance_get_name_ref(         conf_instance              );
//
//
//  // Get time series data from history object and allocate
//  history_alloc_time_series_from_summary_key(history, sum_key, &size, &value, &default_used);
//  std = util_malloc(size * sizeof * std, __func__);
//  
//  {
//    char * error_mode = util_alloc_strupr_copy( __error_mode );
//    // Create  the standard deviation vector
//    if(strcmp(error_mode, "ABS") == 0)
//      {
//	for(int restart_nr = 0; restart_nr < size; restart_nr++)
//	  std[restart_nr] = error;
//      }
//    else if(strcmp(error_mode, "REL") == 0)
//      {
//	for(int restart_nr = 0; restart_nr < size; restart_nr++)
//	  std[restart_nr] = error * abs(value[restart_nr]);
//      }
//    else if(strcmp(error_mode, "RELMIN") == 0)
//      {
//	for(int restart_nr = 0; restart_nr < size; restart_nr++)
//	  {
//	    std[restart_nr] = error * abs(value[restart_nr]);
//	    if(std[restart_nr] < error_min)
//	      std[restart_nr] = error_min;
//	  }
//      }
//    else
//      {
//	util_abort("%s: Internal error. Unknown error mode \"%s\"\n", __func__, __error_mode);
//      }
//    free(error_mode);
//  }
//
//
//  summary_obs_type * summary_obs = summary_obs_alloc(sum_key, size, value, std, default_used);
//  free(value);
//  free(std);
//  free(default_used);
//
//  return summary_obs;
//}
//
//
///**
//   This is used to allocate a summary observation, which is only
//   active at one single instant - i.e. a well test or something like
//   that. The history instance is only used for tranlating between the
//   different time units: report_step, days and dates.
//*/
//
//summary_obs_type * summary_obs_alloc_from_SUMMARY_OBSERVATION(const conf_instance_type * conf_instance , const history_type * history) {
//
//  if(!conf_instance_is_of_class(conf_instance, "SUMMARY_OBSERVATION"))
//    util_abort("%s: internal error. expected \"SUMMARY_OBSERVATION\" instance, got \"%s\".\n",
//               __func__, conf_instance_get_class_name_ref(conf_instance) );
//  
//  int          obs_restart_nr  = 0;
//  double       obs_value       = conf_instance_get_item_value_double(conf_instance, "VALUE" );
//  double       obs_error       = conf_instance_get_item_value_double(conf_instance, "ERROR" );
//  const char * sum_key         = conf_instance_get_item_value_ref(   conf_instance, "KEY"   );
//  const char * obs_key         = conf_instance_get_name_ref(conf_instance);
//  int          size            = history_get_num_restarts(          history          );
//
//  summary_obs_type * summary_obs = NULL;
//
//
//  /** Get the time of the observation. Can be given as date, days or restart. */
//  if(conf_instance_has_item(conf_instance, "RESTART"))
//  {
//    obs_restart_nr = conf_instance_get_item_value_int(conf_instance, "RESTART");
//    if(obs_restart_nr > size)
//      util_abort("%s: Observation %s occurs at restart %i, but history file has only %i restarts.\n",
//                 __func__, obs_key, obs_restart_nr, size);
//  }
//  else if(conf_instance_has_item(conf_instance, "DATE"))
//  {
//    time_t obs_date = conf_instance_get_item_value_time_t(conf_instance, "DATE"  );
//    obs_restart_nr  = history_get_restart_nr_from_time_t(history, obs_date);
//  }
//  else if(conf_instance_has_item(conf_instance, "DAYS"))
//  {
//    double days = conf_instance_get_item_value_double(conf_instance, "DAYS");
//    obs_restart_nr = history_get_restart_nr_from_days(history, days);
//  }
//  else
//    util_abort("%s: Internal error. Invalid conf_instance?\n", __func__);
//
//
//  {
//    double * value        = util_malloc(size * sizeof * value,        __func__);
//    double * std          = util_malloc(size * sizeof * std,          __func__);
//    bool   * default_used = util_malloc(size * sizeof * default_used, __func__);
//
//    /** Fill in defaults .*/
//    for(int restart_nr = 0; restart_nr < size; restart_nr ++)
//    {
//      value       [restart_nr] = 0.0;
//      std         [restart_nr] = 1.0;
//      default_used[restart_nr] = true;
//    }
//
//    value       [obs_restart_nr] = obs_value;
//    std         [obs_restart_nr] = obs_error;
//    default_used[obs_restart_nr] = false;
//    
//    summary_obs = summary_obs_alloc(sum_key, size, value, std, default_used);
//    free(value);
//    free(std);
//    free(default_used);
//  }
//
//  return summary_obs;
//}



/**
   This function want to store the observation for one report step, 
   which is somewhat orthogonal to the design. Currently a no-op.
*/
   
void summary_obs_fread(summary_obs_type * summary_obs , FILE * stream) {

}



/**
   The return value is whether something is actually written to disk. 
*/
bool summary_obs_fwrite(const summary_obs_type * summary_obs , FILE * stream) {
  util_fwrite_double(1.0 , stream);
  util_fwrite_double(1.0 , stream);
  return true;
}


void summary_obs_user_get(const summary_obs_type * summary_obs , const char * index_key , double * value , double * std, bool * valid) {
  *valid = true;
  *value = summary_obs->value;
  *std   = summary_obs->std;
}



/*****************************************************************/

VOID_FREAD(summary_obs)
VOID_FWRITE(summary_obs)
VOID_FREE(summary_obs)
VOID_GET_OBS(summary_obs)
VOID_USER_GET_OBS(summary_obs)
VOID_MEASURE(summary_obs , summary)
