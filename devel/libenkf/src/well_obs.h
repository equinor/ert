#ifndef __WELL_OBS_H__
#define __WELL_OBS_H__
#include <history.h>
#include <enkf_macros.h>
#include <obs_data.h>
#include <meas_vector.h>
#include <well_config.h>
#include <well.h>

typedef enum   {abs_error = 0 , rel_error = 1 , rel_min_abs_error = 2} enkf_obs_error_type; /* Should not have enkf_ prefix */
typedef struct well_obs_struct well_obs_type;

well_obs_type * well_obs_alloc(const well_config_type * , int , const char ** , const history_type * , const double *  , const double *  , const enkf_obs_error_type * );
void            well_obs_free(well_obs_type * );
void            well_obs_get_observations(const well_obs_type *  , int , obs_data_type *);
void            well_obs_measure(const well_obs_type * , const well_type * , meas_vector_type * );
well_obs_type * well_obs_fscanf_alloc(const char * , const well_config_type *  , const history_type * );

VOID_FREE_HEADER(well_obs);
VOID_GET_OBS_HEADER(well_obs);
VOID_MEASURE_HEADER(well);
#endif
