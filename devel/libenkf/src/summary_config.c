#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <enkf_types.h>
#include <enkf_util.h>
#include <summary_config.h>
#include <enkf_macros.h>
#include <util.h>
#include <ecl_well_vars.h>
#include <active_list.h>
#include <ecl_sum.h>
#include <ecl_smspec.h>
#include <set.h>

#define SUMMARY_CONFIG_TYPE_ID 63106

struct summary_config_struct {
  int                   __type_id;
  int                   data_size;
  ecl_smspec_var_type   var_type;    /* The type of the variable - according to ecl_summary nomenclature. */
  char * var;                        /* This is ONE variable of summary.x format - i.e. WOPR:OP_2, RPR:4, ... */
  active_list_type * active_list;    /* overkill with a list here but ... */ 
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
  config->data_size   	       = 1;
  config->active_list          = active_list_alloc( ALL_ACTIVE );
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
  active_list_free(config->active_list);
  set_free(config->obs_set);
  free(config);
}



int summary_config_get_byte_size(const summary_config_type * config) {
  return config->data_size * sizeof(double);
}






/*****************************************************************/
UTIL_SAFE_CAST_FUNCTION(summary_config , SUMMARY_CONFIG_TYPE_ID)
UTIL_SAFE_CAST_FUNCTION_CONST(summary_config , SUMMARY_CONFIG_TYPE_ID)
GET_DATA_SIZE(summary)
VOID_GET_DATA_SIZE(summary)
VOID_CONFIG_FREE(summary)
GET_ACTIVE_LIST(summary)

