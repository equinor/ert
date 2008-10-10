#include <hash.h>
#include <util.h>
#include <cfg.h>


struct cfg_struct
{
  char * type;
  char * name;

  hash_type * sub_keys;  /* A hash indexed with key in the scope. Items  are as specified in the cfg_lang.*/
  hash_type * sub_types; /* A hash indexed with types in the scope. Items are hash'es of cfg_type's indexed by name. */
};


char * cfg_get_string_ref(cfg_type * cfg, const char * key)
{
  if(hash_has_key(cfg->sub_keys, key))
    return (char *) hash_get(cfg->sub_keys, key);
  else{
    util_abort("%s: blah blah\n", __func__);
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



void cfg_alloc_key_from_buffer(cfg_type * cfg, char ** __buffer_pos, const cfg_key_def_type * cfg_key_def)
{


}



cfg_type * cfg_alloc_from_buffer(char ** __buffer, char ** __buffer_pos, cfg_type_def_type * cfg_type_def, const char * name)
{

  cfg_type * cfg = util_malloc(sizeof * cfg, __func__);

  cfg->type = util_alloc_string_copy(cfg_type_def->name);
  cfg->name = util_alloc_string_copy(name);

  hash_alloc(cfg->sub_keys);
  hash_alloc(cfg->sub_types);


  return cfg;
}



