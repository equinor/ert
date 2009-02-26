#ifndef __FIELD_OBS_H__
#define __FIELD_OBS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <conf.h>
#include <history.h>
#include <enkf_macros.h>
#include <obs_data.h>
#include <meas_vector.h>
#include <field_config.h>
#include <field.h>

typedef struct field_obs_struct field_obs_type;

field_obs_type * field_obs_alloc(
  const char   * obs_label,
  const field_config_type * field_config,
  const char   * field_name,
  int            size,
  const int    * i,
  const int    * j,
  const int    * k,
  const double * obs_value,
  const double * obs_std);

void field_obs_free(
  field_obs_type * field_obs);

const char * field_obs_get_field_name(
  const field_obs_type * field_obs);

void field_obs_get_observations(
  const field_obs_type * field_obs,
  int                    restart_nr,
  obs_data_type        * obs_data);

void field_obs_measure(
  const field_obs_type * field_obs,
  const field_type     * field_state,
  meas_vector_type     * meas_vector);

field_obs_type * field_obs_alloc_from_BLOCK_OBSERVATION(
  const conf_instance_type * conf_instance,
  const history_type       * history);




const int * field_obs_get_i(const field_obs_type * );
const int * field_obs_get_j(const field_obs_type * );
const int * field_obs_get_k(const field_obs_type * );
int         field_obs_get_size(const field_obs_type * );
void        field_obs_iget(const field_obs_type * field_obs, int  , double * , double * );

SAFE_CAST_HEADER(field_obs);
VOID_FREE_HEADER(field_obs);
VOID_GET_OBS_HEADER(field_obs);
IS_INSTANCE_HEADER(field_obs);
VOID_MEASURE_HEADER(field_obs);
VOID_USER_GET_OBS_HEADER(field_obs);
VOID_CHI2_HEADER(field_obs);

#ifdef __cplusplus
}
#endif
#endif
