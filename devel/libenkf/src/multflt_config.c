#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <ens_config.h>
#include <multflt_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <trans_func.h>
#include <scalar_config.h>




static multflt_config_type * __multflt_config_alloc_empty(int size) {
  multflt_config_type *multflt_config = malloc(sizeof *multflt_config);
  multflt_config->fault_names   = enkf_util_malloc(size * sizeof *multflt_config->fault_names , __func__);
  multflt_config->scalar_config = scalar_config_alloc_empty(size);

  multflt_config->ecl_kw_name = NULL;
  multflt_config->var_type    = parameter;

  return multflt_config;
}



/*multflt_config_type * multflt_config_alloc(int size, const char * eclfile , const char * ensfile) {
  multflt_config_type *multflt_config = __multflt_config_alloc_empty(size , eclfile , ensfile);
  { 
    int i;
    for (i = 0; i < size; i++) {
      multflt_config->output_transform_name = NULL;
      multflt_config->mean[i]   = 1.0;
      multflt_config->std[i]    = 0.25;
      multflt_config->active[i] = true;
      multflt_config->fault_names[i] = util_alloc_string_copy("FAULT");
      if (multflt_config->active[i])
	multflt_config->serial_size++;
    }
  }

  multflt_config_set_output_transform(multflt_config);
  return multflt_config;
}
*/


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
    if (fscanf(stream , "%s" , name) != 1) {
      fprintf(stderr,"%s: something wrong when reading: %s - aborting \n",__func__ , filename);
      abort();
    }
    config->fault_names[line_nr] = util_alloc_string_copy(name);
    scalar_config_fscanf_line(config->scalar_config , line_nr , stream);
    line_nr++;
  } while ( line_nr < size );
  fclose(stream);
  return config;
}

void multflt_config_free(multflt_config_type * multflt_config) {
  scalar_config_free(multflt_config->scalar_config);
  util_free_string_list(multflt_config->fault_names , scalar_config_get_data_size(multflt_config->scalar_config));
  free(multflt_config);
}


int multflt_config_get_data_size(const multflt_config_type * multflt_config) {
  return scalar_config_get_data_size(multflt_config->scalar_config);
}


const char * multflt_config_get_name(const multflt_config_type * config, int fault_nr) {
  const int size = multflt_config_get_data_size(config);
  if (fault_nr >= 0 && fault_nr < size) 
    return config->fault_names[fault_nr];
  else {
    fprintf(stderr,"%s: asked for fault number:%d - valid interval: [0,%d] - aborting \n",__func__ , fault_nr , size - 1);
    abort();
  }
}


/*****************************************************************/

VOID_FUNC(multflt_config_free , multflt_config_type);
