#ifndef __SUMMARY_OBS_H__
#define __SUMMARY_OBS_H__
#include <enkf_macros.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <summary_config.h>

typedef struct summary_obs_struct summary_obs_type;

summary_obs_type * summary_obs_alloc(const summary_config_type * , const char * , int , const int * , const double *, const double * );
void               summary_obs_fscanf_alloc_data(const char * , int *  , char ***  , double ** , double ** );


VOID_FREE_HEADER(summary_obs);
VOID_GET_OBS_HEADER(summary_obs);
VOID_MEASURE_HEADER(summary_obs);


#endif
