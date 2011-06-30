/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'analysis_config.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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


#ifndef __ANALYSIS_CONFIG_H__
#define __ANALYSIS_CONFIG_H__

#include <config.h>
#include <enkf_types.h>
#include <stdbool.h>

typedef struct analysis_config_struct analysis_config_type;

const char           * analysis_config_get_log_path( const analysis_config_type * config );
void                   analysis_config_init( analysis_config_type * analysis , const config_type * config );
analysis_config_type * analysis_config_alloc_default(void );
void                   analysis_config_free( analysis_config_type * );
bool                   analysis_config_get_merge_observations(const analysis_config_type * );
double                 analysis_config_get_alpha(const analysis_config_type * config);
double                 analysis_config_get_truncation(const analysis_config_type * config);
bool                   analysis_config_Xbased(const analysis_config_type * config);
enkf_mode_type         analysis_config_get_enkf_mode( const analysis_config_type * config );
pseudo_inversion_type  analysis_config_get_inversion_mode( const analysis_config_type * config );
bool                   analysis_config_get_rerun(const analysis_config_type * config);
bool                   analysis_config_get_random_rotation(const analysis_config_type * config);
int                    analysis_config_get_rerun_start(const analysis_config_type * config);
bool                   analysis_config_get_do_local_cross_validation(const analysis_config_type * config);
bool                   analysis_config_get_force_subspace_dimension(const analysis_config_type * config);
bool                   analysis_config_get_do_kernel_regression(const analysis_config_type * config);
int                    analysis_config_get_kernel_function(const analysis_config_type * config);
int                    analysis_config_get_kernel_param(const analysis_config_type * config);
int                    analysis_config_get_nfolds_CV(const analysis_config_type * config);
int                    analysis_config_get_subspace_dimension(const analysis_config_type * config);
bool                   analysis_config_get_bootstrap(const analysis_config_type * config);
bool                   analysis_config_get_penalised_press(const analysis_config_type * config);
bool                   analysis_config_get_do_scaling(const analysis_config_type * config);
void                   analysis_config_set_rerun(analysis_config_type * config , bool rerun);
void                   analysis_config_set_rerun_start( analysis_config_type * config , int rerun_start );
void                   analysis_config_set_truncation( analysis_config_type * config , double truncation);
void                   analysis_config_set_alpha( analysis_config_type * config , double alpha);
void                   analysis_config_set_merge_observations( analysis_config_type * config , bool merge_observations);
void                   analysis_config_set_enkf_mode( analysis_config_type * config , enkf_mode_type enkf_mode);
void                   analysis_config_set_nfolds_CV( analysis_config_type * config , int folds);
void                   analysis_config_set_subspace_dimension( analysis_config_type * config , int dimension);
void                   analysis_config_set_do_bootstrap( analysis_config_type * config , bool do_bootstrap);
void                   analysis_config_set_penalised_press( analysis_config_type * config , bool do_pen_press);
void                   analysis_config_set_log_path(analysis_config_type * config , const char * log_path );
void                   analysis_config_set_std_cutoff( analysis_config_type * config , double std_cutoff );
double                 analysis_config_get_std_cutoff( const analysis_config_type * config );
void                   analysis_config_add_config_items( config_type * config );
void                   analysis_config_fprintf_config( analysis_config_type * config , FILE * stream);

#endif
