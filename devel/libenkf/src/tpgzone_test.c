#include <tpgzone_config.h>

int main()
{
  hash_type * empty_hash = hash_alloc();
  
  tpgzone_config_type * config;

  config = tpgzone_config_fscanf_alloc("TEST.CONF",NULL);

  tpgzone_config_free(config);

  hash_free(empty_hash);
};
