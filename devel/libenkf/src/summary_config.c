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





//int summary_config_get_var_index(const summary_config_type * config , const char * var) {
//  int index , i;
//  index = -1;
//  i = 0;
//  while (index < 0 && i < config->data_size) {
//    if (strcmp(var , config->var_list[i]) == 0) index = i;
//    i++;
//  }
//  return index;
//}
//
//
//
//bool summary_config_has_var(const summary_config_type * config , const char * var) {
//  int index;
//  index = summary_config_get_var_index(config , var);
//  if (index >= 0)
//    return true;
//  else
//    return false;
//}
//
//
//const char ** summary_config_get_var_list_ref(const summary_config_type * config) { return (const char **) config->var_list; }
//
//
//void summary_config_add_var(summary_config_type * config , const char * var) {
//  if (summary_config_has_var(config , var)) {
//    fprintf(stderr,"%s: summary variable:%s already added to summary object - nothing done \n",__func__ , var);
//    return;
//  }
//  
//  if (true /* variable valid */) {
//    config->data_size++;
//    active_list_grow(config->active_list , 1);
//    config->var_list = realloc(config->var_list , config->data_size * sizeof * config->var_list);
//    config->var_list[config->data_size - 1] = util_alloc_string_copy(var);
//  } else {
//    fprintf(stderr,"%s: summary variable: %s not recognized - aborting \n",__func__ , var);
//    abort();
//  }
//}




//static summary_config_type * __summary_config_alloc( void ) {
//  summary_config_type * config = malloc(sizeof *config);
//
//  config->data_size   	  = 0;
//  config->var_list    	  = NULL;
//  config->active_list     = active_list_alloc(0);
//  return config;
//
//}



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

