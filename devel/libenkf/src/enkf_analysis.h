#ifndef __ENKF_ANALYSIS_H__
#define __ENKF_ANALYSIS_H__

#include <matrix.h>
#include <obs_data.h>
#include <analysis_config.h>
#include <int_vector.h>
#include <mzran.h>

matrix_type * enkf_analysis_allocX( const analysis_config_type * config , mzran_type * rng , const meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot);
matrix_type * enkf_analysis_allocX_cv( const analysis_config_type * config , mzran_type * rng , meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot , matrix_type * A);
matrix_type * enkf_analysis_allocX_pre_cv( const analysis_config_type * config , mzran_type * rng , meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot , matrix_type * A , matrix_type * V0T , matrix_type * Z , double * eig , matrix_type * U0 , meas_data_type * fasit , int unique_bootstrap_components);
matrix_type * enkf_analysis_allocX_boot( const analysis_config_type * config , mzran_type * rng , const meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot , const meas_data_type * fasit);
void          enkf_analysis_local_pre_cv( const analysis_config_type * config , mzran_type * rng , meas_data_type * meas_data , obs_data_type * obs_data , matrix_type  * V0T , matrix_type * Z , double * eig , matrix_type * U0 , meas_data_type * fasit);
void          enkf_analysis_fprintf_obs_summary(const obs_data_type * obs_data , const meas_data_type * meas_data  , const int_vector_type * step_list , const char * ministep_name ,  FILE * stream );
void          enkf_analysis_deactivate_outliers(obs_data_type * obs_data , meas_data_type * meas_data , double std_cutoff , double alpha);
matrix_type * enkf_analysis_alloc_mp_randrot(int ens_size , mzran_type * rng);

#endif
