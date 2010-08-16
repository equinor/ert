#ifndef __SUMMARY_OBS_H__
#define __SUMMARY_OBS_H__

#ifdef __cplusplus 
extern "C" {
#endif


#include <stdbool.h>
#include <conf.h>
#include <history.h>
#include <enkf_macros.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <summary_config.h>
#include <summary.h>
#include <active_list.h>
#include <util.h>

typedef struct summary_obs_struct summary_obs_type;

void summary_obs_free(
  summary_obs_type * summary_obs);

summary_obs_type * summary_obs_alloc(
  const char   * summary_key,
  const char * obs_key , 
  double  value ,
  double  std);


bool summary_obs_default_used(
  const summary_obs_type * summary_obs,
  int                      restart_nr);

const char * summary_obs_get_summary_key(
  const summary_obs_type * summary_obs);

summary_obs_type * summary_obs_alloc_from_HISTORY_OBSERVATION(
  const conf_instance_type * conf_instance,
  const history_type       * history);

summary_obs_type * summary_obs_alloc_from_SUMMARY_OBSERVATION(
  const conf_instance_type * conf_instance,
  const history_type       * history);

void summary_obs_set(summary_obs_type * , double , double );


VOID_FREE_HEADER(summary_obs);
VOID_GET_OBS_HEADER(summary_obs);
VOID_MEASURE_HEADER(summary_obs);
UTIL_IS_INSTANCE_HEADER(summary_obs);
VOID_USER_GET_OBS_HEADER(summary_obs);
VOID_CHI2_HEADER(summary_obs);

#ifdef __cplusplus
}
#endif
#endif
