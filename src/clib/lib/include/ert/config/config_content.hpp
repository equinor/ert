#ifndef ERT_CONFIG_CONTENT_H
#define ERT_CONFIG_CONTENT_H

#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/stringlist.hpp>
#include <ert/util/vector.hpp>

#include <ert/config/config_content_item.hpp>
#include <ert/config/config_path_elm.hpp>
#include <ert/config/config_path_stack.hpp>
#include <ert/config/config_schema_item.hpp>

namespace fs = std::filesystem;

struct config_content_struct {
    /** A set of config files which have been parsed - to protect against
     * circular includes. */
    std::set<std::string> parsed_files;
    vector_type *nodes;
    hash_type *items;
    std::vector<std::string> parse_errors;
    stringlist_type *warnings;
    subst_list_type *define_list;
    char *config_file;
    char *abs_path;
    char *config_path;

    config_path_stack_type *path_stack;
    /** Absolute path to directory that contains current config */
    fs::path invoke_path;
    bool valid;
};

typedef struct config_content_struct config_content_type;

extern "C" config_content_type *config_content_alloc(const char *filename);
extern "C" void config_content_free(config_content_type *content);
void config_content_set_valid(config_content_type *content);
extern "C" bool config_content_is_valid(const config_content_type *content);
extern "C" bool config_content_has_item(const config_content_type *content,
                                        const char *key);
void config_content_add_item(config_content_type *content,
                             const config_schema_item_type *schema_item,
                             const config_path_elm_type *path_elm);
extern "C" config_content_item_type *
config_content_get_item(const config_content_type *content, const char *key);
void config_content_add_node(config_content_type *content,
                             config_content_node_type *content_node);

const char *config_content_iget(const config_content_type *content,
                                const char *key, int occurence, int index);
int config_content_get_occurences(const config_content_type *content,
                                  const char *kw);
const char *
config_content_get_value_as_abspath(const config_content_type *config,
                                    const char *kw);
const char *config_content_get_value(const config_content_type *config,
                                     const char *kw);
config_content_node_type *
config_content_get_value_node(const config_content_type *content,
                              const char *kw);
extern "C" void config_content_add_define(config_content_type *content,
                                          const char *key, const char *value);
subst_list_type *config_content_get_define_list(config_content_type *content);
extern "C" const subst_list_type *
config_content_get_const_define_list(const config_content_type *content);
const char *config_content_get_config_file(const config_content_type *content,
                                           bool abs_path);
extern "C" int config_content_get_size(const config_content_type *content);
extern "C" const config_content_node_type *
config_content_iget_node(const config_content_type *content, int index);
bool config_content_add_file(config_content_type *content,
                             const char *config_file);
extern "C" config_path_elm_type *
config_content_add_path_elm(config_content_type *content, const char *path);
extern "C" const stringlist_type *
config_content_get_warnings(const config_content_type *content);
extern "C" const char *
config_content_get_config_path(const config_content_type *content);
void config_content_pop_path_stack(config_content_type *content);
extern "C" stringlist_type *
config_content_alloc_keys(const config_content_type *content);

#endif
