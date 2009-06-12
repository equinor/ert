#ifndef __ANALYSIS_H__
#define __ANALYSIS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <meas_matrix.h>
#include <obs_data.h>
#include <enkf_types.h>
#include <config.h>
#include <analysis_config.h>


void     analysis_set_stride(int , int , int * , int * );
double * old_analysis_allocX(int , int , const meas_matrix_type * , obs_data_type * , bool , bool , const analysis_config_type *);


#ifdef __cplusplus
}
#endif
#endif
