/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'analysis_config.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <config.h>
#include <util.h>
#include <enkf_types.h>
#include <analysis_config.h>
#include <enkf_defaults.h>
#include <rng.h>
#include "config_keys.h"


struct analysis_config_struct {
  bool                    merge_observations;  /* When observing from time1 to time2 - should ALL observations in between be used? */
  bool                    rerun;               /* Should we rerun the simulator when the parameters have been updated? */
  int                     rerun_start;         /* When rerunning - from where should we start? */
  bool                    random_rotation;     /* When using the SQRT scheme - should a random rotation be performed?? Fixed to TRUE*/
  double                  truncation;
  double                  overlap_alpha;
  double                  std_cutoff;
  enkf_mode_type          enkf_mode;
  pseudo_inversion_type   inversion_mode;            /* Fixed to SVD_SS_N1_R */ 
  char                  * log_path;                  /* Points to directory with update logs. */
  int                     nfolds_CV;                 /* Number of folds in the CV scheme */
  bool                    do_local_cross_validation; /* Should we do CV for separate state vector matrices? */
  bool                    bootstrap;              /* Should we do bootstrapping?*/
}; 









void analysis_config_set_std_cutoff( analysis_config_type * config , double std_cutoff ) {
  config->std_cutoff = std_cutoff;
}


void analysis_config_set_log_path(analysis_config_type * config , const char * log_path ) {
  config->log_path        = util_realloc_string_copy(config->log_path , log_path);
}

/**
   Will in addition create the path.
*/
const char * analysis_config_get_log_path( const analysis_config_type * config ) {
  util_make_path( config->log_path );
  return config->log_path; 
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

double analysis_config_get_std_cutoff(const analysis_config_type * config) {
  return config->std_cutoff;
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

int analysis_config_get_nfolds_CV(const analysis_config_type * config) {
  return config->nfolds_CV;
}


bool analysis_config_get_do_local_cross_validation(const analysis_config_type * config) {
  return config->do_local_cross_validation;
}

bool analysis_config_get_bootstrap(const analysis_config_type * config) {
  return config->bootstrap;
}

static void analysis_config_set_bootstrap(analysis_config_type * config , bool bootstrap) {
  config->bootstrap = bootstrap;
}

static void analysis_config_set_CV(analysis_config_type * config , bool CV) {
  config->do_local_cross_validation = CV;
}

void analysis_config_set_truncation( analysis_config_type * config , double truncation) {
  config->truncation = truncation;
}

void analysis_config_set_alpha( analysis_config_type * config , double alpha) {
  config->overlap_alpha = alpha;
}

void analysis_config_set_merge_observations( analysis_config_type * config , bool merge_observations) {
  config->merge_observations = merge_observations;
}

void analysis_config_set_enkf_mode( analysis_config_type * config , enkf_mode_type enkf_mode) {
  config->enkf_mode = enkf_mode;
}

void analysis_config_set_nfolds_CV( analysis_config_type * config , int folds) {
  config->nfolds_CV = folds;
}


static const char * analysis_config_get_mode_string( enkf_mode_type mode ) {
  switch( mode ) {
  case( ENKF_STANDARD ):
    return "STANDARD";
    break;
  case( ENKF_SQRT ):
    return "SQRT";
    break;
  default:
    util_abort("%s: fallen off the end of a switch ?\n",__func__);
    return NULL;
  }
}

/**
   The analysis_config object is instantiated with the default values
   for enkf_defaults.h
*/


void analysis_config_init( analysis_config_type * analysis , const config_type * config ) {
  if (config_item_set( config , UPDATE_LOG_PATH_KEY ))
    analysis_config_set_log_path( analysis , config_get_value( config , UPDATE_LOG_PATH_KEY ));
  
  if (config_item_set( config , STD_CUTOFF_KEY ))
    analysis_config_set_std_cutoff( analysis , config_get_value_as_double( config , STD_CUTOFF_KEY ));
  
  if (config_item_set( config , ENKF_TRUNCATION_KEY ))
    analysis_config_set_truncation( analysis , config_get_value_as_double( config , ENKF_TRUNCATION_KEY ));
  
  if (config_item_set( config , ENKF_ALPHA_KEY ))
    analysis_config_set_alpha( analysis , config_get_value_as_double( config , ENKF_ALPHA_KEY ));

  if (config_item_set( config , ENKF_MERGE_OBSERVATIONS_KEY ))
    analysis_config_set_merge_observations( analysis , config_get_value_as_bool( config , ENKF_MERGE_OBSERVATIONS_KEY ));

  if (config_item_set( config , ENKF_MODE_KEY )) {
    const char * enkf_mode_string = config_get_value(config , ENKF_MODE_KEY);
    enkf_mode_type enkf_mode      = ENKF_SQRT; /* Compiler shut up */
    
    if (strcmp(enkf_mode_string,"STANDARD") == 0)
      enkf_mode = ENKF_STANDARD;
    else if (strcmp(enkf_mode_string , "SQRT") == 0)
      enkf_mode = ENKF_SQRT;
    else
      util_abort("%s: internal error : enkf_mode:%s not recognized \n",__func__ , enkf_mode_string);
  
    analysis_config_set_enkf_mode( analysis , enkf_mode );
  }
  
  if (config_item_set( config , ENKF_RERUN_KEY ))
    analysis_config_set_rerun( analysis , config_get_value_as_bool( config , ENKF_RERUN_KEY ));

  if (config_item_set( config , RERUN_START_KEY ))
    analysis_config_set_rerun_start( analysis , config_get_value_as_int( config , RERUN_START_KEY ));
  
  if (config_item_set( config , ENKF_LOCAL_CV_KEY )) {
    analysis_config_set_CV( analysis , config_get_value_as_bool(config , ENKF_LOCAL_CV_KEY ));

    if (config_item_set( config , ENKF_CV_FOLDS_KEY ))
      analysis_config_set_nfolds_CV( analysis , config_get_value_as_int( config , ENKF_CV_FOLDS_KEY ));
  } 
  
  /*Bootstrap parameters: */  
  if (config_item_set( config , ENKF_BOOTSTRAP_KEY)) 
    analysis_config_set_bootstrap( analysis , config_get_value_as_bool( config , ENKF_BOOTSTRAP_KEY ));

}



bool analysis_config_get_merge_observations(const analysis_config_type * config) {
  return config->merge_observations;
}



double analysis_config_get_alpha(const analysis_config_type * config) {
  return config->overlap_alpha;
}


double analysis_config_get_truncation(const analysis_config_type * config) {
  return config->truncation;
}



void analysis_config_free(analysis_config_type * config) {
  free(config->log_path);
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


analysis_config_type * analysis_config_alloc_default() {
  analysis_config_type * config = util_malloc( sizeof * config , __func__);
  
  config->inversion_mode            = SVD_SS_N1_R;
  config->random_rotation           = true;
  config->log_path                  = NULL;

  analysis_config_set_CV( config                 , DEFAULT_ENKF_CV);
  analysis_config_set_bootstrap( config          , DEFAULT_ENKF_BOOTSTRAP );
  analysis_config_set_std_cutoff( config         , DEFAULT_ENKF_STD_CUTOFF );
  analysis_config_set_log_path( config           , DEFAULT_UPDATE_LOG_PATH );
  analysis_config_set_truncation( config         , DEFAULT_ENKF_TRUNCATION );
  analysis_config_set_alpha( config              , DEFAULT_ENKF_ALPHA );
  analysis_config_set_merge_observations( config , DEFAULT_MERGE_OBSERVATIONS );
  analysis_config_set_enkf_mode ( config         , DEFAULT_ENKF_MODE );
  analysis_config_set_rerun( config              , DEFAULT_RERUN );
  analysis_config_set_rerun_start( config        , DEFAULT_RERUN_START );
  analysis_config_set_nfolds_CV( config          , DEFAULT_CV_NFOLDS );         
  
  return config;
}



/*****************************************************************/
/*
  Keywords for the analysis - all optional. The analysis_config object
  is instantiated with defaults from enkf_defaults.h
*/

void analysis_config_add_config_items( config_type * config ) {
  config_item_type * item;
  
  item = config_add_key_value(config , ENKF_MODE_KEY , false , CONFIG_STRING );
  config_item_set_common_selection_set(item , 2 , (const char *[2]) {"STANDARD" , "SQRT"});
  
  config_add_key_value( config , STD_CUTOFF_KEY              , false , CONFIG_FLOAT);
  config_add_key_value( config , ENKF_TRUNCATION_KEY         , false , CONFIG_FLOAT);
  config_add_key_value( config , ENKF_ALPHA_KEY              , false , CONFIG_FLOAT);
  config_add_key_value( config , ENKF_MERGE_OBSERVATIONS_KEY , false , CONFIG_BOOLEAN);
  config_add_key_value( config , ENKF_CROSS_VALIDATION_KEY   , false , CONFIG_BOOLEAN);
  config_add_key_value( config , ENKF_LOCAL_CV_KEY           , false , CONFIG_BOOLEAN);
  config_add_key_value( config , ENKF_BOOTSTRAP_KEY          , false , CONFIG_BOOLEAN);
  config_add_key_value( config , ENKF_CV_FOLDS_KEY           , false , CONFIG_INT);
  config_add_key_value( config , ENKF_RERUN_KEY              , false , CONFIG_BOOLEAN);
  config_add_key_value( config , RERUN_START_KEY             , false , CONFIG_INT);
  config_add_key_value( config , UPDATE_LOG_PATH_KEY         , false , CONFIG_STRING);
}



void analysis_config_fprintf_config( analysis_config_type * config , FILE * stream) {
  fprintf( stream , CONFIG_COMMENTLINE_FORMAT );
  fprintf( stream , CONFIG_COMMENT_FORMAT , "Here comes configuration information related to the EnKF analysis.");

  if (config->std_cutoff != DEFAULT_ENKF_STD_CUTOFF) {
    fprintf( stream , CONFIG_KEY_FORMAT   , STD_CUTOFF_KEY );
    fprintf( stream , CONFIG_FLOAT_FORMAT , config->std_cutoff );
    fprintf( stream , "\n");
  }

  if (config->truncation != DEFAULT_ENKF_TRUNCATION ) {
    fprintf( stream , CONFIG_KEY_FORMAT   , ENKF_TRUNCATION_KEY );
    fprintf( stream , CONFIG_FLOAT_FORMAT , config->truncation );
    fprintf( stream , "\n");
  }

  if (config->overlap_alpha != DEFAULT_ENKF_ALPHA ) {
    fprintf( stream , CONFIG_KEY_FORMAT   , ENKF_TRUNCATION_KEY );
    fprintf( stream , CONFIG_FLOAT_FORMAT , config->overlap_alpha );
    fprintf( stream , "\n");
  }
  
  if (config->merge_observations != DEFAULT_MERGE_OBSERVATIONS) {
    fprintf( stream , CONFIG_KEY_FORMAT        , ENKF_MERGE_OBSERVATIONS_KEY);
    fprintf( stream , CONFIG_ENDVALUE_FORMAT   , CONFIG_BOOL_STRING( config->merge_observations ));
  }

  if (config->rerun) {
    fprintf( stream , CONFIG_KEY_FORMAT        , ENKF_RERUN_KEY);
    fprintf( stream , CONFIG_ENDVALUE_FORMAT   , CONFIG_BOOL_STRING( config->rerun ));
  }
  
  if (config->rerun_start != DEFAULT_RERUN_START) {
    fprintf( stream , CONFIG_KEY_FORMAT   , RERUN_START_KEY);
    fprintf( stream , CONFIG_INT_FORMAT   , config->rerun_start );
    fprintf( stream , "\n");
  }

  if (config->enkf_mode != DEFAULT_ENKF_MODE) {
    fprintf( stream , CONFIG_KEY_FORMAT      , ENKF_MODE_KEY);
    fprintf( stream , CONFIG_ENDVALUE_FORMAT , analysis_config_get_mode_string( config->enkf_mode ));
  }
  
  if (config->log_path != NULL) {
    fprintf( stream , CONFIG_KEY_FORMAT      , UPDATE_LOG_PATH_KEY);
    fprintf( stream , CONFIG_ENDVALUE_FORMAT , config->log_path );
  }
 
  if (config->do_local_cross_validation) {
    fprintf( stream , CONFIG_KEY_FORMAT        , ENKF_LOCAL_CV_KEY );
    fprintf( stream , CONFIG_ENDVALUE_FORMAT   , CONFIG_BOOL_STRING( config->do_local_cross_validation ));
  }

  if (config->nfolds_CV != DEFAULT_CV_NFOLDS ) {
    fprintf( stream , CONFIG_KEY_FORMAT   , ENKF_CV_FOLDS_KEY );
    fprintf( stream , CONFIG_INT_FORMAT   , config->nfolds_CV );
    fprintf( stream , "\n");
  }

  fprintf(stream , "\n\n");
}



