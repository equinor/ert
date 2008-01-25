#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <enkf_types.h>
#include <enkf_util.h>
#include <well_config.h>
#include <enkf_macros.h>
#include <util.h>
#include <ecl_well_vars.h>


/*****************************************************************/


int well_config_get_var_index(const well_config_type * config , const char * var) {
  int index , i;
  index = -1;
  i = 0;
  while (index < 0 && i < config->data_size) {
    if (strcmp(var , config->var_list[i]) == 0) index = i;
    i++;
  }
  return index;
}



bool well_config_has_var(const well_config_type * config , const char * var) {
  int index;
  index = well_config_get_var_index(config , var);
  if (index >= 0)
    return true;
  else
    return false;
}


const char ** well_config_get_var_list_ref(const well_config_type * config) { return (const char **) config->var_list; }


void well_config_add_var(well_config_type * config , const char * var) {
  if (well_config_has_var(config , var)) {
    fprintf(stderr,"%s: well variable:%s already added to well:%s - nothing done \n",__func__ , var , config->well_name);
    return;
  }
  /*
    All variables are (currently) active.
  */
  if (ecl_well_var_valid(var , NULL)) {
    config->data_size++;
    config->var_list = realloc(config->var_list , config->data_size * sizeof * config->var_list);
    config->var_list[config->data_size - 1] = util_alloc_string_copy(var);
  } else {
    fprintf(stderr,"%s: well variable: %s not recognized - aborting \n",__func__ , var);
    abort();
  }
}


static well_config_type * __well_config_alloc(const char * well_name) {
  well_config_type * config = malloc(sizeof *config);

  config->well_name   = util_alloc_string_copy(well_name);
  config->data_size   = 0;
  config->var_list    = NULL;
  return config;
}



well_config_type * well_config_alloc(const char * well_name , int size , const char ** var_list) {
  well_config_type * config = __well_config_alloc(well_name);
  int i;
  for (i=0; i < size; i++) 
    well_config_add_var(config , var_list[i]);
  
  return config;
}



well_config_type * well_config_fscanf_alloc(const char * well_name , const char * filename) {
  FILE * stream = util_fopen(filename , "r");
  char * line   = NULL;
  int ivar;
  int size      = util_count_file_lines(stream);
  bool at_eof;

  fseek(stream , 0L , SEEK_SET);
  well_config_type * config = __well_config_alloc(well_name);
  for (ivar = 0; ivar < size; ivar++) {
    char var[32];
    line = util_fscanf_realloc_line(stream , &at_eof , line);
    if (!at_eof) {
      if (sscanf(line , "%s" , var) == 1) 
	well_config_add_var(config , var);
      else {
	fprintf(stderr,"%s: line:%s not recognized as valid format - aborting \n",__func__ , line);
	abort();
      }
    }
  } 
  
  free(line);
  fclose(stream);
  return config;
}



void well_config_free(well_config_type * config) {
  int i;
  for (i = 0; i < config->data_size; i++)
    free(config->var_list[i]);
  free(config->var_list);
  free(config);
}



const char * well_config_get_well_name_ref(const well_config_type * config) {
  return config->well_name;
}




/*****************************************************************/
GET_DATA_SIZE(well)
VOID_CONFIG_FREE(well)


