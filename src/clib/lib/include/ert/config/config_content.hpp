#ifndef ERT_CONFIG_CONTENT_H
#define ERT_CONFIG_CONTENT_H

#include <ert/res_util/subst_list.hpp>
#include <ert/util/stringlist.hpp>
#include <ert/util/type_macros.hpp>

#include <ert/config/config_content_item.hpp>
#include <ert/config/config_error.hpp>
#include <ert/config/config_schema_item.hpp>

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
extern "C" config_error_type *
config_content_get_errors(const config_content_type *content);

const char *config_content_iget(const config_content_type *content,
                                const char *key, int occurence, int index);
extern "C" int config_content_iget_as_int(const config_content_type *content,
                                          const char *key, int occurence,
                                          int index);
bool config_content_iget_as_bool(const config_content_type *content,
                                 const char *key, int occurence, int index);
int config_content_get_occurences(const config_content_type *content,
                                  const char *kw);

bool config_content_get_value_as_bool(const config_content_type *config,
                                      const char *kw);
int config_content_get_value_as_int(const config_content_type *config,
                                    const char *kw);
double config_content_get_value_as_double(const config_content_type *config,
                                          const char *kw);
const char *config_content_get_value_as_path(const config_content_type *config,
                                             const char *kw);
const char *
config_content_get_value_as_abspath(const config_content_type *config,
                                    const char *kw);
const char *
config_content_get_value_as_executable(const config_content_type *config,
                                       const char *kw);
const char *config_content_get_value(const config_content_type *config,
                                     const char *kw);
const stringlist_type *
config_content_iget_stringlist_ref(const config_content_type *content,
                                   const char *kw, int occurence);
config_content_node_type *
config_content_get_value_node(const config_content_type *content,
                              const char *kw);
extern "C" void config_content_add_define(config_content_type *content,
                                          const char *key, const char *value);
subst_list_type *config_content_get_define_list(config_content_type *content);
const subst_list_type *
config_content_get_const_define_list(const config_content_type *content);
const char *config_content_get_config_file(const config_content_type *content,
                                           bool abs_path);
extern "C" int config_content_get_size(const config_content_type *content);
const config_content_node_type *
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

UTIL_IS_INSTANCE_HEADER(config_content);

#endif
