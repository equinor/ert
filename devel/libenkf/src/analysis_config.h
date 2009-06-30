#ifndef __ANALYSIS_CONFIG_H__
#define __ANALYSIS_CONFIG_H__

#include <config.h>
#include <enkf_types.h>
#include <stdbool.h>

typedef struct analysis_config_struct analysis_config_type;

analysis_config_type * analysis_config_alloc(const config_type * );
void                   analysis_config_free( analysis_config_type * );
bool                   analysis_config_merge_observations(const analysis_config_type * );
double 		       analysis_config_get_alpha(const analysis_config_type * config);
double 		       analysis_config_get_truncation(const analysis_config_type * config);
int    		       analysis_config_get_fortran_enkf_mode(const analysis_config_type * config);
bool                   analysis_config_Xbased(const analysis_config_type * config);
enkf_mode_type         analysis_config_get_enkf_mode( const analysis_config_type * config );
pseudo_inversion_type  analysis_config_get_inversion_mode( const analysis_config_type * config );
double                 analysis_config_get_std_cutoff(const analysis_config_type * config);
bool                   analysis_config_get_rerun(const analysis_config_type * config);
int                    analysis_config_get_rerun_start(const analysis_config_type * config);
void                   analysis_config_set_rerun(analysis_config_type * config , bool rerun);
void                   analysis_config_set_rerun_start( analysis_config_type * config , int rerun_start );



#endif
