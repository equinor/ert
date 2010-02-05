#ifndef __ENKF_ANALYSIS_H__
#define __ENKF_ANALYSIS_H__

#include <matrix.h>
#include <obs_data.h>
#include <analysis_config.h>

matrix_type * enkf_analysis_allocX( const analysis_config_type * config , meas_matrix_type * meas_matrix , obs_data_type * obs_data , const matrix_type * randrot);
void 	      enkf_analysis_fprintf_obs_summary(const obs_data_type * obs_data , const meas_matrix_type * meas_matrix  , int start_step , int end_step , const char * ministep_name ,  FILE * stream );
void 	      enkf_analysis_deactivate_outliers(obs_data_type * obs_data , meas_matrix_type * meas_matrix , double std_cutoff , double alpha);
matrix_type * enkf_analysis_alloc_mp_randrot(int ens_size );

#endif
