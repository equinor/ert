#ifndef __GEN_OBS_H__
#define __GEN_OBS_H__
#include <gen_data_config.h>
#include <enkf_macros.h>
#include <meas_vector.h>
#include <gen_data_config.h>
#include <obs_data.h>

typedef struct gen_obs_struct gen_obs_type;

gen_obs_type * gen_obs_alloc( const char * obs_key , const char * , double , double , const char * , const char * );
void           gen_obs_user_get_with_data_index(const gen_obs_type * gen_obs , const char * index_key , double * value , double * std , bool * valid);

VOID_CHI2_HEADER(gen_obs);
UTIL_IS_INSTANCE_HEADER(gen_obs);
VOID_FREE_HEADER(gen_obs);
VOID_GET_OBS_HEADER(gen_obs);
VOID_MEASURE_HEADER(gen_obs);
VOID_USER_GET_OBS_HEADER(gen_obs);

#endif
