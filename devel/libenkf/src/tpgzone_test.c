#include <tpgzone_config.h>

int main()
{
  hash_type * empty_hash = hash_alloc();
  
  tpgzone_config_type * config;

  config = tpgzone_config_fscanf_alloc("TEST.CONF",NULL);

  printf("%i\n",hash_key_list_compare(config->facies_kw_hash,config->facies_kw_hash));
  printf("%i\n",hash_key_list_compare(config->facies_kw_hash,empty_hash));

  tpgzone_config_free(config);

  hash_free(empty_hash);
};
