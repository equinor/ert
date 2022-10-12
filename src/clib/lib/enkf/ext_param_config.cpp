#include <stdlib.h>

#include <algorithm>
#include <string>
#include <vector>

#include <ert/util/util.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/ext_param_config.hpp>

struct ext_param_config_struct {
    std::string key;
    std::vector<std::string> keys;
    std::vector<std::vector<std::string>> suffixes;
};

void ext_param_config_free(ext_param_config_type *config) { delete config; }

int ext_param_config_get_data_size(const ext_param_config_type *config) {
    return config->keys.size();
}

const char *ext_param_config_iget_key(const ext_param_config_type *config,
                                      int index) {
    return config->keys[index].data();
}

int ext_param_config_get_key_index(const ext_param_config_type *config,
                                   const char *key) {
    const auto it = std::find(config->keys.begin(), config->keys.end(), key);
    return it == config->keys.end() ? -1
                                    : std::distance(config->keys.begin(), it);
}

bool ext_param_config_has_key(const ext_param_config_type *config,
                              const char *key) {
    return std::find(config->keys.begin(), config->keys.end(), key) !=
           config->keys.end();
}

ext_param_config_type *ext_param_config_alloc(const char *key,
                                              const stringlist_type *keys) {
    if (stringlist_get_size(keys) == 0)
        return NULL;

    if (!stringlist_unique(keys))
        return NULL;

    ext_param_config_type *config = new ext_param_config_type();
    config->key = key;

    for (int i = 0; i < stringlist_get_size(keys); i++) {
        config->keys.push_back(stringlist_iget(keys, i));
    }
    config->suffixes.resize(stringlist_get_size(keys));
    return config;
}

void ext_param_config_ikey_set_suffixes(ext_param_config_type *config, int ikey,
                                        const stringlist_type *suffixes) {
    auto tmp = std::vector<std::string>(stringlist_get_size(suffixes));
    for (int isuffix = 0; isuffix < stringlist_get_size(suffixes); isuffix++)
        tmp[isuffix] = stringlist_iget(suffixes, isuffix);
    config->suffixes[ikey] = std::move(tmp);
}

int ext_param_config_ikey_get_suffix_count(const ext_param_config_type *config,
                                           int ikey) {
    return config->suffixes[ikey].size();
}

const char *
ext_param_config_ikey_iget_suffix(const ext_param_config_type *config, int ikey,
                                  int isuffix) {
    return config->suffixes[ikey][isuffix].data();
}

int ext_param_config_ikey_get_suffix_index(const ext_param_config_type *config,
                                           int ikey, const char *suffix) {
    const auto it = std::find(config->suffixes[ikey].begin(),
                              config->suffixes[ikey].end(), suffix);
    return it == config->suffixes[ikey].end()
               ? -1
               : std::distance(config->suffixes[ikey].begin(), it);
}

VOID_FREE(ext_param_config)
VOID_GET_DATA_SIZE(ext_param)
