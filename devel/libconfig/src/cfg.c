#include <hash.h>
#include <util.h>
#include <cfg.h>


struct cfg_struct
{
  char * type;
  char * name;

  hash_type * sub_keys;  /* A hash indexed with key in the scope. Items will be as specified in the cfg_lang, or parsing will fail.*/
  hash_type * sub_types; /* A hash indexed with types in the scope. Items are hash'es of cfg_type's indexed by name. */
};



char * cfg_get_key_string_ref(cfg_type * cfg, const char * key)
{
  if(hash_has_key(cfg->sub_keys, key))
    return (char *) hash_get(cfg->sub_keys, key);
  else{
    util_abort("%s: Key %s has not been set in the current scope.\n", __func__, key);
    return NULL;
  }
}



void cfg_free(cfg_type * cfg)
{
  free(cfg->type);
  free(cfg->name);

  hash_free(cfg->sub_keys);
  hash_free(cfg->sub_types);
}



void cfg_free__(void * cfg)
{
  cfg_free( (cfg_type *) cfg);
}



