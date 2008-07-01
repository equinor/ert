#include <stdlib.h>
#include <util.h>
#include <gen_data.h>
#include <gen_data_config.h>




int main(int argc , char **argv) {
  int i;
  gen_data_config_type * config = gen_data_config_fscanf_alloc("gen_config.txt");
  gen_data_type * gen_data      = gen_data_alloc(config);
  
  for (i= 0; i < 100; i++)
    gen_data_ecl_load(gen_data , "./" , NULL , NULL , i);
  
  gen_data_free(gen_data);
  gen_data_config_free(config);
}
