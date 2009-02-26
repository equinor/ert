#ifndef __ANALYSIS_H__
#define __ANALYSIS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <meas_matrix.h>
#include <obs_data.h>
#include <enkf_types.h>
#include <config.h>

typedef struct analysis_config_struct analysis_config_type;

void     analysis_set_stride(int , int , int * , int * );
double * analysis_allocX(int , int , const meas_matrix_type * , obs_data_type * , bool , bool , const analysis_config_type *);

analysis_config_type * analysis_config_alloc(const config_type * );
void                   analysis_config_free( analysis_config_type * );
bool                   analysis_config_merge_observations(const analysis_config_type * );

#ifdef __cplusplus
}
#endif
#endif
