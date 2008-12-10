#ifndef __GEN_OBS_H__
#define __GEN_OBS_H__
#include <gen_data_config.h>
#include <enkf_macros.h>
#include <meas_vector.h>
#include <gen_data_config.h>
#include <gen_obs_active.h>
#include <obs_data.h>

typedef struct gen_obs_struct gen_obs_type;

gen_obs_type * gen_obs_alloc( const char * );

SAFE_CAST_HEADER(gen_obs);
IS_INSTANCE_HEADER(gen_obs);
VOID_OBS_ACTIVATE_HEADER(gen_obs)
VOID_FREE_HEADER(gen_obs);
VOID_GET_OBS_HEADER(gen_obs);
VOID_MEASURE_HEADER(gen_obs);
VOID_USER_GET_OBS_HEADER(gen_obs);

#endif
