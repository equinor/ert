#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <util.h>
#include <ens_config.h>
#include <enkf_macros.h>
#include <relperm_config.h>
#include <util.h>
#include <trans_func.h>


static relperm_config_type * __relperm_config_alloc_empty( int size ) {
  
  relperm_config_type *relperm_config = malloc(sizeof *relperm_config);

  relperm_config->scalar_config = scalar_config_alloc_empty(size);
  relperm_config->var_type = parameter;

  relperm_config->relperm_num1 = enkf_util_malloc(size * sizeof  *relperm_config->relperm_num1, __func__);
  relperm_config->relperm_num2 = enkf_util_malloc(size * sizeof  *relperm_config->relperm_num2, __func__);
  return relperm_config;
}



relperm_config_type * relperm_config_fscanf_alloc(const char * filename){
  relperm_config_type * config;
  FILE * stream = util_fopen(filename, "r");
  int size, line_nr;
  int num1;
  int num2;

  size = util_count_file_lines(stream);
  fseek(stream, 0L, SEEK_SET);
  config = __relperm_config_alloc_empty(size);
  line_nr = 0;
  
  do {    
    if (fscanf(stream,"%i %i" , &num1 , &num2) != 2) {
      printf("%s Something wrong with the input in relperm_config \n",__func__);
      abort();
    }
    config->relperm_num1[line_nr] = num1;
    config->relperm_num2[line_nr] = num2;

    scalar_config_fscanf_line(config->scalar_config, line_nr , stream);
    printf("relperm_num1 %d \n",config->relperm_num1[line_nr]);
    printf("relperm_num2 %d \n",config->relperm_num2[line_nr]);
    
    
    line_nr++;
  }while(line_nr < size);

  fclose(stream);
  return config;
}

int relperm_config_get_data_size(const relperm_config_type * relperm_config) {
  return scalar_config_get_data_size(relperm_config->scalar_config);
}

void relperm_config_ecl_write(const relperm_config_type * config, const double * data, FILE * stream){
  int ik;
  for (ik = 0 ; ik < relperm_config_get_data_size(config);ik++){
    fprintf(stream,"RELPERM\n %d %d %g \n",config->relperm_num1[ik],config->relperm_num2[ik],data[ik]);
  }
}
