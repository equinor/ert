#ifndef ERT_CONFIG_PATH_ELM_H
#define ERT_CONFIG_PATH_ELM_H

#include <filesystem>

struct config_path_elm_type {
    /** UTIL_TYPE_ID_DECLARATION */
    int __type_id;
    std::filesystem::path path;
};

extern "C" void config_path_elm_free(config_path_elm_type *path_elm);
void config_path_elm_free__(void *arg);
config_path_elm_type *
config_path_elm_alloc(const std::filesystem::path &root_path, const char *path);
extern "C" const char *
config_path_elm_get_abspath(const config_path_elm_type *path_elm);
char *config_path_elm_alloc_path(const config_path_elm_type *path_elm,
                                 const char *input_path);
#define config_path_elm_alloc_abspath(path_elm, input_path)                    \
    config_path_elm_alloc_path(path_elm, input_path)

#endif
