#include <stdlib.h>

#include <filesystem>
#include <set>
#include <string>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/type_macros.hpp>
#include <ert/util/vector.hpp>

#include <ert/config/config_content.hpp>
#include <ert/config/config_path_elm.hpp>
#include <ert/config/config_path_stack.hpp>

namespace fs = std::filesystem;

#define CONFIG_CONTENT_TYPE_ID 6612520

struct config_content_struct {
    UTIL_TYPE_ID_DECLARATION;
    /** A set of config files which have been parsed - to protect against
     * circular includes. */
    std::set<std::string> parsed_files;
    vector_type *nodes;
    hash_type *items;
    config_error_type *parse_errors;
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

UTIL_IS_INSTANCE_FUNCTION(config_content, CONFIG_CONTENT_TYPE_ID)

config_content_type *config_content_alloc(const char *filename) {
    auto content = new config_content_type;
    UTIL_TYPE_ID_INIT(content, CONFIG_CONTENT_TYPE_ID);

    content->valid = false;
    content->items = hash_alloc();
    content->nodes = vector_alloc_new();
    content->parse_errors = config_error_alloc();
    content->define_list = subst_list_alloc(NULL);
    content->warnings = stringlist_alloc_new();

    content->path_stack = config_path_stack_alloc();
    content->config_file = util_alloc_string_copy(filename);
    content->abs_path = util_alloc_abs_path(filename);
    content->config_path = util_split_alloc_dirname(content->abs_path);
    content->invoke_path = fs::current_path();

    return content;
}

bool config_content_has_item(const config_content_type *content,
                             const char *key) {
    return hash_has_key(content->items, key);
}

config_content_item_type *
config_content_get_item(const config_content_type *content, const char *key) {
    return (config_content_item_type *)hash_get(content->items, key);
}

void config_content_add_item(config_content_type *content,
                             const config_schema_item_type *schema_item,
                             const config_path_elm_type *path_elm) {

    const char *kw = config_schema_item_get_kw(schema_item);
    config_content_item_type *content_item =
        config_content_item_alloc(schema_item, path_elm);
    hash_insert_hash_owned_ref(content->items, kw, content_item,
                               config_content_item_free__);

    if (config_schema_item_is_deprecated(schema_item))
        stringlist_append_copy(
            content->warnings,
            config_schema_item_get_deprecate_msg(schema_item));
}

void config_content_add_node(config_content_type *content,
                             config_content_node_type *content_node) {
    vector_append_ref(content->nodes, content_node);
}

void config_content_set_valid(config_content_type *content) {
    content->valid = true;
}

bool config_content_is_valid(const config_content_type *content) {
    return content->valid;
}

config_error_type *
config_content_get_errors(const config_content_type *content) {
    return content->parse_errors;
}

const stringlist_type *
config_content_get_warnings(const config_content_type *content) {
    return content->warnings;
}

void config_content_free(config_content_type *content) {
    if (!content)
        return;

    stringlist_free(content->warnings);
    vector_free(content->nodes);
    hash_free(content->items);
    config_error_free(content->parse_errors);
    subst_list_free(content->define_list);
    free(content->config_file);
    free(content->abs_path);
    free(content->config_path);

    config_path_stack_free(content->path_stack);
    delete content;
}

bool config_content_add_file(config_content_type *content,
                             const char *config_file) {
    const auto iter = content->parsed_files.find(config_file);
    if (iter == content->parsed_files.end()) {
        content->parsed_files.insert(config_file);
        return true;
    }
    return false;
}

/*
   Here comes some xxx_get() functions - many of them will fail if
   the item has not been added in the right way (this is to ensure that
   the xxx_get() request is unambigous.
*/

/*
   This function can be used to get the value of a config
   parameter. But to ensure that the get is unambigous we set the
   following requirements to the item corresponding to 'kw':

    * argc_minmax has been set to 1,1

   If this is not the case - we die.
*/

/*
   Assume we installed a key 'KEY' which occurs three times in the final
   config file:

   KEY  1    2     3
   KEY  11   22    33
   KEY  111  222   333


   Now when accessing these values the occurence variable will
   correspond to the linenumber, and the index will index along a line:

     config_iget_as_int( config , "KEY" , 0 , 2) => 3
     config_iget_as_int( config , "KEY" , 2 , 1) => 222
*/

const char *config_content_iget(const config_content_type *content,
                                const char *key, int occurence, int index) {
    config_content_item_type *item = config_content_get_item(content, key);
    return config_content_item_iget(item, occurence, index);
}

int config_content_iget_as_int(const config_content_type *content,
                               const char *key, int occurence, int index) {
    config_content_item_type *item = config_content_get_item(content, key);
    return config_content_item_iget_as_int(item, occurence, index);
}

bool config_content_iget_as_bool(const config_content_type *content,
                                 const char *key, int occurence, int index) {
    config_content_item_type *item = config_content_get_item(content, key);
    return config_content_item_iget_as_bool(item, occurence, index);
}

/**
   Return the number of times a keyword has been set - dies on unknown
   'kw'. If the append_arg attribute has been set to false the
   function will return 0 or 1 irrespective of how many times the item
   has been set in the config file.
*/
int config_content_get_occurences(const config_content_type *content,
                                  const char *kw) {
    if (config_content_has_item(content, kw))
        return config_content_item_get_size(
            config_content_get_item(content, kw));
    else
        return 0;
}

const config_content_node_type *
config_content_iget_node(const config_content_type *content, int index) {
    const config_content_node_type *node =
        (const config_content_node_type *)vector_iget_const(content->nodes,
                                                            index);
    return node;
}

int config_content_get_size(const config_content_type *content) {
    return vector_get_size(content->nodes);
}

/* All the get_value functions will operate on the last item which has
   been set with a particular key value. So assuming the config file
   looks like:

   KEY   VALUE1
   KEY   VALUE2  OPTIONAL
   KEY   100     VALUE3   OPTIONAL  ERROR

   these functions will all operate on the last line in the config file:

             KEY 100 VALUE3 OPTIONAL ERROR
*/

static config_content_node_type *
config_content_get_value_node__(const config_content_type *config,
                                const char *kw) {
    config_content_node_type *node = config_content_get_value_node(config, kw);
    if (node == NULL)
        util_abort("Tried to get value node from unset kw:%s \n", __func__, kw);

    return node;
}

config_content_node_type *
config_content_get_value_node(const config_content_type *content,
                              const char *kw) {
    config_content_item_type *item = config_content_get_item(content, kw);
    config_content_node_type *node = config_content_item_get_last_node(item);
    config_content_node_assert_key_value(node);
    return node;
}

bool config_content_get_value_as_bool(const config_content_type *config,
                                      const char *kw) {
    config_content_node_type *node =
        config_content_get_value_node__(config, kw);
    return config_content_node_iget_as_bool(node, 0);
}

int config_content_get_value_as_int(const config_content_type *config,
                                    const char *kw) {
    config_content_node_type *node =
        config_content_get_value_node__(config, kw);
    return config_content_node_iget_as_int(node, 0);
}

double config_content_get_value_as_double(const config_content_type *config,
                                          const char *kw) {
    config_content_node_type *node =
        config_content_get_value_node__(config, kw);
    return config_content_node_iget_as_double(node, 0);
}

const char *config_content_get_value_as_path(const config_content_type *config,
                                             const char *kw) {
    config_content_node_type *node =
        config_content_get_value_node__(config, kw);
    return config_content_node_iget_as_path(node, 0);
}

const char *
config_content_get_value_as_abspath(const config_content_type *config,
                                    const char *kw) {
    config_content_node_type *node =
        config_content_get_value_node__(config, kw);
    return config_content_node_iget_as_abspath(node, 0);
}

const char *
config_content_get_value_as_executable(const config_content_type *config,
                                       const char *kw) {
    config_content_node_type *node =
        config_content_get_value_node__(config, kw);
    return config_content_node_iget_as_executable(node, 0);
}

const char *config_content_get_value(const config_content_type *config,
                                     const char *kw) {
    config_content_node_type *node =
        config_content_get_value_node__(config, kw);
    return config_content_node_iget(node, 0);
}

const stringlist_type *
config_content_iget_stringlist_ref(const config_content_type *content,
                                   const char *kw, int occurence) {
    config_content_item_type *item = config_content_get_item(content, kw);

    return config_content_item_iget_stringlist_ref(item, occurence);
}

void config_content_add_define(config_content_type *content, const char *key,
                               const char *value) {
    char *filtered_value =
        subst_list_alloc_filtered_string(content->define_list, value);
    subst_list_append_copy(content->define_list, key, filtered_value, NULL);
    free(filtered_value);
}

subst_list_type *config_content_get_define_list(config_content_type *content) {
    return content->define_list;
}

const subst_list_type *
config_content_get_const_define_list(const config_content_type *content) {
    return content->define_list;
}

const char *config_content_get_config_file(const config_content_type *content,
                                           bool abs_path) {
    if (abs_path)
        return content->abs_path;
    else
        return content->config_file;
}

config_path_elm_type *config_content_add_path_elm(config_content_type *content,
                                                  const char *path) {
    const config_path_elm_type *current_path_elm{};

    if (config_path_stack_size(content->path_stack) == 0)
        current_path_elm = NULL;
    else
        current_path_elm = config_path_stack_get_last(content->path_stack);

    config_path_elm_type *new_path_elm;
    const auto &invoke_path = current_path_elm == nullptr
                                  ? content->invoke_path
                                  : current_path_elm->path;
    if (path != NULL) {
        auto new_path = fs::absolute(invoke_path / path);
        new_path_elm = config_path_elm_alloc(invoke_path, new_path.c_str());
    } else {
        new_path_elm = config_path_elm_alloc(invoke_path, nullptr);
    }
    config_path_stack_append(content->path_stack, new_path_elm);
    return new_path_elm;
}

const char *config_content_get_config_path(const config_content_type *content) {
    return content->config_path;
}

void config_content_pop_path_stack(config_content_type *content) {
    config_path_stack_pop(content->path_stack);
}

stringlist_type *config_content_alloc_keys(const config_content_type *content) {
    return hash_alloc_stringlist(content->items);
}
