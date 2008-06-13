#include <tpgzone_config.h>

int main()
{
  hash_type * empty_hash = hash_alloc();
  
  tpgzone_config_type * config;

  config = tpgzone_config_fscanf_alloc("TEST.CONF",NULL);


  printf("The number of target fields is %i\n",config->num_target_fields);
  {
    int i;
    for(i=0; i<config->num_target_fields; i++)
      printf(" - Target field %02i is %s\n",i,config->target_keys[i]);

  }

  printf("\nHave loaded petrophysics for the following facies:\n");
  hash_printf_keys(config->facies_kw_hash);


  printf("Testing petrophysics, hang on to your helmet!\n");
  {
    int i;
    int j;

    for(i=0; i< config->num_target_fields; i++)
    {
      for(j=0; j<config->num_facies; j++)
      printf("%f\n",scalar_config_transform_item(config->petrophysics[i],0.0,j));
    }
  }

  printf("The number of Gaussian fields is %i\n",config->num_gauss_fields);

  if(config->petrophysics == NULL) printf("Something weird..\n");

  if(config->petrophysics[0] == NULL) printf("Something weird..\n");


  tpgzone_config_free(config);

  hash_free(empty_hash);
};
