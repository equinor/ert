#ifndef __SUMMARY_OBS_H__
#define __SUMMARY_OBS_H__
#include <stdbool.h>
#include <conf.h>
#include <history.h>
#include <enkf_macros.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <summary_config.h>
#include <summary.h>

typedef struct summary_obs_struct summary_obs_type;

void summary_obs_free(
  summary_obs_type * summary_obs);

summary_obs_type * summary_obs_alloc(
  const char   * summary_key,
  int            size,
  const double * value ,
  const double * std,
  const bool   * default_used);

bool summary_obs_default_used(
  const summary_obs_type * summary_obs,
  int                      restart_nr);

const char * summary_obs_get_summary_key_ref(
  const summary_obs_type * summary_obs);

void summary_obs_get_observations(
  const summary_obs_type * summary_obs,
  int                      restart_nr,
  obs_data_type          * obs_data);

void summary_obs_measure(
  const summary_obs_type * obs,
  const summary_type     * summary,
  meas_vector_type       * meas_vector);

summary_obs_type * summary_obs_alloc_from_HISTORY_OBSERVATION(
  const conf_instance_type * conf_instance,
  const history_type       * history);

summary_obs_type * summary_obs_alloc_from_SUMMARY_OBSERVATION(
  const conf_instance_type * conf_instance,
  const history_type       * history);



VOID_FREE_HEADER(summary_obs);
VOID_GET_OBS_HEADER(summary_obs);
VOID_MEASURE_HEADER(summary_obs);


#endif
