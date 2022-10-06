#ifndef EXT_PARAM_CONFIG_H
#define EXT_PARAM_CONFIG_H
#include <ert/enkf/enkf_macros.hpp>

typedef struct ext_param_config_struct ext_param_config_type;

extern "C" ext_param_config_type *
ext_param_config_alloc(const char *key, const stringlist_type *keys);

extern "C" void ext_param_config_free(ext_param_config_type *config);
extern "C" void
ext_param_config_ikey_set_suffixes(ext_param_config_type *config, int ikey,
                                   const stringlist_type *suffixes);

extern "C" int
ext_param_config_get_data_size(const ext_param_config_type *config);
extern "C" const char *
ext_param_config_iget_key(const ext_param_config_type *config, int index);
extern "C" int
ext_param_config_get_key_index(const ext_param_config_type *config,
                               const char *key);
extern "C" bool ext_param_config_has_key(const ext_param_config_type *config,
                                         const char *key);
extern "C" int
ext_param_config_ikey_get_suffix_count(const ext_param_config_type *config,
                                       int key_id);
extern "C" const char *
ext_param_config_ikey_iget_suffix(const ext_param_config_type *config,
                                  int key_id, int suffix_id);
int ext_param_config_ikey_get_suffix_index(const ext_param_config_type *config,
                                           int key_id, const char *suffix);

UTIL_SAFE_CAST_HEADER(ext_param_config);
UTIL_SAFE_CAST_HEADER_CONST(ext_param_config);
VOID_FREE_HEADER(ext_param_config);
VOID_GET_DATA_SIZE_HEADER(ext_param);

#endif
