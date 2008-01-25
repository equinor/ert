#ifndef __FIELD_OBS_H__
#define __FIELD_OBS_H__
#include <history.h>
#include <enkf_macros.h>
#include <obs_data.h>
#include <meas_vector.h>
#include <field_config.h>
#include <field.h>

typedef struct field_obs_struct field_obs_type;

field_obs_type * field_obs_alloc(const field_config_type *  , const char *  , int , const int * , const int *, const int * , const double *);
void             field_obs_free(field_obs_type * );
void             field_obs_get_observations(const field_obs_type *  , int , obs_data_type *);
void             field_obs_measure(const field_obs_type * , const field_type * , meas_vector_type * );
field_obs_type * field_obs_fscanf_alloc(const char * , const field_config_type *  );

VOID_FREE_HEADER(field_obs);
VOID_GET_OBS_HEADER(field_obs);
VOID_MEASURE_HEADER(field);
#endif
