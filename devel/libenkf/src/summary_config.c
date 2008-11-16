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


struct summary_config_struct {
  CONFIG_STD_FIELDS;
  char * var;                    /* This is ONE variable of summary.x format - i.e. WOPR:OP_2, RPR:4, ... */
  active_list_type * active_list;/* overkill with a list here but ... */ 
};


/*****************************************************************/



const char * summary_config_get_var(const summary_config_type * config) {
  return config->var;
}


summary_config_type * summary_config_alloc(const char * var) {
  summary_config_type * config = malloc(sizeof *config);
  config->data_size   	  = 1;
  config->active_list     = active_list_alloc(1);
  config->var             = util_alloc_string_copy( var );

  return config;
}




void summary_config_free(summary_config_type * config) {
  free(config->var);
  active_list_free(config->active_list);
  free(config);
}







/*****************************************************************/
GET_DATA_SIZE(summary)
VOID_CONFIG_FREE(summary)
GET_ACTIVE_LIST(summary)

