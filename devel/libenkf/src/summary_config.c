/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'summary_config.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <enkf_types.h>
#include <enkf_util.h>
#include <summary_config.h>
#include <enkf_macros.h>
#include <util.h>
#include <ecl_sum.h>
#include <ecl_smspec.h>
#include <set.h>
#include <summary_config.h>

#define SUMMARY_CONFIG_TYPE_ID 63106

struct summary_config_struct {
  int                   __type_id;
  ecl_smspec_var_type   var_type;    /* The type of the variable - according to ecl_summary nomenclature. */
  char * var;                        /* This is ONE variable of summary.x format - i.e. WOPR:OP_2, RPR:4, ... */
  set_type           *  obs_set;     /* Set of keys (which fit in enkf_obs) which are observations of this node. */ 
};


/*****************************************************************/



const char * summary_config_get_var(const summary_config_type * config) {
  return config->var;
}


ecl_smspec_var_type summary_config_get_var_type(summary_config_type * config , const ecl_sum_type * ecl_sum) {
  /* This might race as hell - but what the fuck. */
  if (config->var_type == ECL_SMSPEC_INVALID_VAR) 
    config->var_type = ecl_sum_identify_var_type( ecl_sum , config->var );


  return config->var_type;
}


summary_config_type * summary_config_alloc(const char * var) {
  summary_config_type * config = util_malloc(sizeof *config , __func__);
  config->var                  = util_alloc_string_copy( var );
  config->var_type             = ECL_SMSPEC_INVALID_VAR;
  config->obs_set              = set_alloc_empty(); 
  config->__type_id            = SUMMARY_CONFIG_TYPE_ID;
  return config;
}


void summary_config_add_obs_key(summary_config_type * config, const char * obs_key) {
  set_add_key(config->obs_set , obs_key);
}



void summary_config_free(summary_config_type * config) {
  free(config->var);
  set_free(config->obs_set);
  free(config);
}



int summary_config_get_byte_size(const summary_config_type * config) {
  return sizeof(double);
}


int summary_config_get_data_size( const summary_config_type * config) {
  return 1;
}






/*****************************************************************/
UTIL_SAFE_CAST_FUNCTION(summary_config , SUMMARY_CONFIG_TYPE_ID)
UTIL_SAFE_CAST_FUNCTION_CONST(summary_config , SUMMARY_CONFIG_TYPE_ID)
VOID_GET_DATA_SIZE(summary)
VOID_CONFIG_FREE(summary)

