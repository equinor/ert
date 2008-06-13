#include <tpgzone_config.h>

int main()
{
  hash_type * empty_hash = hash_alloc();
  
  tpgzone_config_type * config;

  config = tpgzone_config_fscanf_alloc("TEST.CONF",NULL);

  {
    int i;
    for(i=0; i<config->num_target_fields; i++)
      printf("Target field %02i is %s\n",i,config->target_keys[i]);

  }

  if(config->petrophysics == NULL) printf("Something weird..\n");

  if(config->petrophysics[0] == NULL) printf("Something weird..\n");


  tpgzone_config_free(config);

  hash_free(empty_hash);
};
