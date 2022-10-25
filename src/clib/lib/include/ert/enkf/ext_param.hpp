#ifndef EXT_PARAM_H
#define EXT_PARAM_H

#include <ert/enkf/ext_param_config.hpp>

typedef struct ext_param_struct ext_param_type;

extern "C" bool ext_param_iset(ext_param_type *param, int index, double value);
bool ext_param_iiset(ext_param_type *param, int index, int suffix_index,
                     double value);
extern "C" double ext_param_iget(const ext_param_type *param, int index);
double ext_param_iiget(const ext_param_type *param, int index,
                       int suffix_index);
extern "C" bool ext_param_key_set(ext_param_type *param, const char *key,
                                  double value);
extern "C" bool ext_param_key_suffix_set(ext_param_type *param, const char *key,
                                         const char *suffix, double value);
extern "C" double ext_param_key_get(const ext_param_type *param,
                                    const char *key);
extern "C" double ext_param_key_suffix_get(const ext_param_type *param,
                                           const char *key, const char *suffix);
extern "C" void ext_param_json_export(const ext_param_type *ext_param,
                                      const char *json_file);
extern "C" void ext_param_free(ext_param_type *ext_param);
extern "C" ext_param_type *ext_param_alloc(const ext_param_config_type *config);
extern "C" ext_param_config_type const *
ext_param_get_config(const ext_param_type *param);

VOID_FREE_HEADER(ext_param);
VOID_ALLOC_HEADER(ext_param);
VOID_WRITE_TO_BUFFER_HEADER(ext_param)
VOID_READ_FROM_BUFFER_HEADER(ext_param)

#endif
