#ifndef __ENKF_ANALYSIS_H__
#define __ENKF_ANALYSIS_H__

#include <matrix.h>
#include <obs_data.h>
#include <analysis_config.h>

matrix_type * enkf_analysis_allocX( const analysis_config_type * config , const meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot);
matrix_type * enkf_analysis_allocX_cv( const analysis_config_type * config , meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot , matrix_type * A);
matrix_type * enkf_analysis_allocX_pre_cv( const analysis_config_type * config , meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot , matrix_type * A , matrix_type * V0T , matrix_type * Z , double * eig , matrix_type * U0);
void          enkf_analysis_local_pre_cv( const analysis_config_type * config , meas_data_type * meas_data , obs_data_type * obs_data , matrix_type  * V0T , matrix_type * Z , double * eig , matrix_type * U0);
void          enkf_analysis_fprintf_obs_summary(const obs_data_type * obs_data , const meas_data_type * meas_data  , int start_step , int end_step , const char * ministep_name ,  FILE * stream );
void          enkf_analysis_deactivate_outliers(obs_data_type * obs_data , meas_data_type * meas_data , double std_cutoff , double alpha);
matrix_type * enkf_analysis_alloc_mp_randrot(int ens_size );

#endif
