#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <ens_config.h>
#include <multflt_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <trans_func.h>
#include <scalar_config.h>


#define MULTFLT_CONFIG_ID 881563

struct multflt_config_struct {
  int                   __type_id;
  char                * ecl_kw_name;      
  enkf_var_type         var_type;  
  scalar_config_type  * scalar_config;
  char               ** fault_names;
};



static multflt_config_type * __multflt_config_alloc_empty(int size) {
  multflt_config_type *config   = util_malloc(sizeof * config , __func__);
  config->__type_id     = MULTFLT_CONFIG_ID;
  config->fault_names   = util_malloc(size * sizeof * config->fault_names , __func__);
  config->scalar_config = scalar_config_alloc_empty(size);

  config->ecl_kw_name = NULL;
  config->var_type    = parameter;

  return config;
}

SAFE_CAST(multflt_config , MULTFLT_CONFIG_ID)



void multflt_config_transform(const multflt_config_type * config , const double * input_data , double * output_data) {
  scalar_config_transform(config->scalar_config , input_data , output_data);
}



multflt_config_type * multflt_config_fscanf_alloc(const char * filename ) {
  multflt_config_type * config;
  FILE * stream = util_fopen(filename , "r");
  int line_nr = 0;
  int size;

  size = util_count_file_lines(stream);
  fseek(stream , 0L , SEEK_SET);
  config = __multflt_config_alloc_empty(size);
  do {
    char name[128];  /* UGGLY HARD CODED LIMIT */
    if (fscanf(stream , "%s" , name) != 1) 
      util_abort("%s: something wrong when reading: %s - aborting \n",__func__ , filename);
    
    config->fault_names[line_nr] = util_alloc_string_copy(name);
    scalar_config_fscanf_line(config->scalar_config , line_nr , stream);
    line_nr++;
  } while ( line_nr < size );
  fclose(stream);
  return config;
}

void multflt_config_free(multflt_config_type * multflt_config) {
  util_free_stringlist(multflt_config->fault_names , scalar_config_get_data_size(multflt_config->scalar_config));
  scalar_config_free(multflt_config->scalar_config);
  free(multflt_config);
}


int multflt_config_get_data_size(const multflt_config_type * multflt_config) {
  return scalar_config_get_data_size(multflt_config->scalar_config);
}


const char ** multflt_config_get_names(const multflt_config_type * config) {
  return (const char **) config->fault_names;
}


const char * multflt_config_get_name(const multflt_config_type * config, int fault_nr) {
  const int size = multflt_config_get_data_size(config);
  if (fault_nr >= 0 && fault_nr < size) 
    return config->fault_names[fault_nr];
  else {
    util_abort("%s: asked for fault number:%d - valid interval: [0,%d] - aborting \n",__func__ , fault_nr , size - 1);
    return NULL;  /* Keep the compiler happy */
  }
}


void multflt_config_activate(multflt_config_type * config , active_mode_type active_mode , void * active_config) {
  /*
   */
}


scalar_config_type * multflt_config_get_scalar_config( const multflt_config_type * config) {
  return config->scalar_config;
}



/**
   Will return -1 if the index is invalid.
*/
int multflt_config_get_index(const multflt_config_type * config , const char * fault_name) {
  const int size   = multflt_config_get_data_size(config);
  bool    have_fault = false;
  int     index    = 0;
  
  while (index < size && !have_fault) {
    if (strcmp(config->fault_names[index] , fault_name) == 0)
      have_fault = true;
    else
      index++;
  }
  
  if (have_fault)
    return index;
  else
    return -1;
}



/*****************************************************************/
VOID_FREE(multflt_config)
VOID_CONFIG_ACTIVATE(multflt)
