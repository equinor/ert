/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'enkf_analysis.h' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/


#ifndef __ENKF_ANALYSIS_H__
#define __ENKF_ANALYSIS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <matrix.h>
#include <obs_data.h>
#include <analysis_config.h>
#include <int_vector.h>
#include <rng.h>

matrix_type * enkf_analysis_allocX_pre_cv(const analysis_config_type * config , 
                                          rng_type * rng , 
                                          meas_data_type * meas_data , 
                                          obs_data_type * obs_data , 
                                          const matrix_type * randrot , 
                                          const matrix_type * A , 
                                          const matrix_type * V0T , 
                                          const matrix_type * Z , 
                                          const double * eig , 
                                          const matrix_type * U0 , 
                                          meas_data_type * fasit , 
                                          int unique_bootstrap_components);

matrix_type * enkf_analysis_allocX_boot( const analysis_config_type * config , 
                                         rng_type * rng , 
                                         const meas_data_type * meas_data , 
                                         obs_data_type * obs_data , 
                                         const matrix_type * randrot , 
                                         const meas_data_type * fasit);


void enkf_analysis_initX_principal_components_cv( int nfolds_CV , 
                                                  bool penalised_press , 
                                                  matrix_type * X , 
                                                  rng_type * rng , 
                                                  const matrix_type * A , 
                                                  const matrix_type * Z , 
                                                  const matrix_type * Rp , 
                                                  const matrix_type * Dp );


void enkf_analysis_local_pre_cv( const analysis_config_type * config , 
                                 rng_type * rng , 
                                 meas_data_type * meas_data , 
                                 obs_data_type * obs_data , 
                                 matrix_type  * V0T , 
                                 matrix_type * Z , 
                                 double * eig , 
                                 matrix_type * U0 , 
                                 meas_data_type * fasit);

//void enkf_analysis_alloc_matrices( rng_type * rng , 
//                                   const meas_data_type * meas_data , 
//                                   obs_data_type * obs_data , 
//                                   enkf_mode_type enkf_mode , 
//                                   matrix_type ** S , 
//                                   matrix_type ** R , 
//                                   matrix_type ** innov,
//                                   matrix_type ** E ,
//                                   matrix_type ** D , 
//                                   bool scale);

void          enkf_analysis_init_principal_components( double truncation , 
                                                       const matrix_type * S, 
                                                       const matrix_type * R,
                                                       const matrix_type * innov,
                                                       const matrix_type * E , 
                                                       const matrix_type * D , 
                                                       matrix_type * Z , 
                                                       matrix_type * Rp, 
                                                       matrix_type * Dp);

void          enkf_analysis_fprintf_obs_summary(const obs_data_type * obs_data , 
                                                const meas_data_type * meas_data  , 
                                                const int_vector_type * step_list , 
                                                const char * ministep_name ,
                                                FILE * stream );
  
void          enkf_analysis_deactivate_outliers(obs_data_type * obs_data , 
                                                meas_data_type * meas_data , 
                                                double std_cutoff , 
                                                double alpha);

matrix_type * enkf_analysis_alloc_mp_randrot(int ens_size , rng_type * rng);


#ifdef __cplusplus
}
#endif

#endif
