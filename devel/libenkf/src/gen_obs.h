#ifndef __GEN_OBS_H__
#define __GEN_OBS_H__
#include <gen_data_config.h>
#include <enkf_macros.h>
#include <enkf_obs.h>
#include <obs_node.h>
#include <meas_vector.h>
#include <gen_data_config.h>
#include <gen_obs_active.h>

typedef struct gen_obs_struct gen_obs_type;

gen_obs_type * gen_obs_alloc( const char * );

VOID_OBS_ACTIVATE_HEADER(gen_obs)
VOID_FREE_HEADER(gen_obs);
VOID_GET_OBS_HEADER(gen_obs);
VOID_MEASURE_HEADER(gen_obs);


#endif
