#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <config.h>
#include <util.h>
#include <enkf_types.h>
#include <analysis_config.h>




struct analysis_config_struct {
  bool                   merge_observations;  /* When observing from time1 to time2 - should ALL observations in between be used? */
  bool                   rerun;               /* Should we rerun the simulator when the parameters have been updated? */
  int                    rerun_start;         /* When rerunning - from where should we start? */
  bool                   random_rotation;     /* When using the SQRT scheme - should a random rotation be performed?? */
  double 	         truncation;
  double 	         overlap_alpha;
  double                 std_cutoff;
  enkf_mode_type         enkf_mode;
  pseudo_inversion_type  inversion_mode;
  int                    fortran_enkf_mode; 
};



static analysis_config_type * analysis_config_alloc__(double truncation , double overlap_alpha , enkf_mode_type enkf_mode , bool merge_observations) {
  analysis_config_type * config = util_malloc( sizeof * config , __func__);

  config->merge_observations = merge_observations;
  config->truncation         = truncation;
  config->overlap_alpha      = overlap_alpha;
  config->enkf_mode          = enkf_mode;
  config->inversion_mode     = SVD_SS_N1_R;
  config->std_cutoff         = 1e-6;
  config->random_rotation    = true;
  
  config->fortran_enkf_mode  = config->enkf_mode + config->inversion_mode;
  return config;
}


void analysis_config_set_rerun(analysis_config_type * config , bool rerun) {
  config->rerun = rerun;
}


void analysis_config_set_rerun_start( analysis_config_type * config , int rerun_start ) {
  config->rerun_start = rerun_start;
}


bool analysis_config_get_rerun(const analysis_config_type * config) {
  return config->rerun;
}


int analysis_config_get_rerun_start(const analysis_config_type * config) {
  return config->rerun_start;
}


bool analysis_config_get_random_rotation(const analysis_config_type * config) {
  if (config->enkf_mode == ENKF_SQRT)
    return config->random_rotation;
  else
    return false;
}


analysis_config_type * analysis_config_alloc(const config_type * config) {
  double truncation 	    = config_iget_as_double(config , "ENKF_TRUNCATION" , 0,0);
  double alpha      	    = config_iget_as_double(config , "ENKF_ALPHA" , 0,0);
  bool   merge_observations = config_iget_as_bool(config , "ENKF_MERGE_OBSERVATIONS" , 0,0);
  const char * enkf_mode_string = config_iget(config , "ENKF_MODE" , 0,0);
  enkf_mode_type enkf_mode = ENKF_SQRT; /* Compiler shut up */
  
  if (strcmp(enkf_mode_string,"STANDARD") == 0)
    enkf_mode = ENKF_STANDARD;
  else if (strcmp(enkf_mode_string , "SQRT") == 0)
    enkf_mode = ENKF_SQRT;
  else
    util_abort("%s: internal error : enkf_mode:%s not recognized \n",__func__ , enkf_mode_string);
  
  {
    analysis_config_type * analysis = analysis_config_alloc__(truncation , alpha , enkf_mode , merge_observations);
    analysis_config_set_rerun( analysis , config_iget_as_bool(config , "ENKF_RERUN", 0,0));
    analysis_config_set_rerun_start( analysis , config_iget_as_int( config , "RERUN_START" ,0,0));
    return analysis;
  }
}


bool analysis_config_merge_observations(const analysis_config_type * config) {
  return config->merge_observations;
}

double analysis_config_get_alpha(const analysis_config_type * config) {
  return config->overlap_alpha;
}

double analysis_config_get_std_cutoff(const analysis_config_type * config) {
  return config->std_cutoff;
}

double analysis_config_get_truncation(const analysis_config_type * config) {
  return config->truncation;
}

int analysis_config_get_fortran_enkf_mode(const analysis_config_type * config) {
  return config->fortran_enkf_mode;
}


void analysis_config_free(analysis_config_type * config) {
  free(config);
}


enkf_mode_type analysis_config_get_enkf_mode( const analysis_config_type * config ) {
  return config->enkf_mode;
}


pseudo_inversion_type analysis_config_get_inversion_mode( const analysis_config_type * config ) {
  return config->inversion_mode;
}


bool analysis_config_Xbased(const analysis_config_type * config) {
  if ((config->enkf_mode == ENKF_STANDARD) || (config->enkf_mode == ENKF_SQRT))
    return true;
  else
    return false;
}

