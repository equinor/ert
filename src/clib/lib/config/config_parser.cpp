#include <filesystem>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ert/logging.hpp>
#include <ert/res_util/res_env.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/parser.hpp>
#include <ert/util/path_stack.hpp>
#include <optional>

#include <ert/config/config_parser.hpp>

static auto logger = ert::get_logger("config");

namespace fs = std::filesystem;

#define CLEAR_STRING "__RESET__"

/**
Structure to parse configuration files of this type:

KEYWORD1  ARG2   ARG2  ARG3
KEYWORD2  ARG1-2
....
KEYWORDN  ARG1  ARG2

A keyword can occure many times.


                           =============================
                           | config_type object        |
                           |                           |
                           | Contains 'all' the        |
                           | configuration information.|
                           |                           |
                           =============================
                               |                   |
                               |                   \________________________
                               |                                            \
                              KEY1                                         KEY2
                               |                                             |
                              \|/                                           \|/
                   =========================                      =========================
                   | config_item object    |                      | config_item object    |
                   |                       |                      |                       |
                   | Indexed by a keyword  |                      | Indexed by a keyword  |
                   | which is the first    |                      | which is the first    |
                   | string in the         |                      | string in the         |
                   | config file.          |                      | config file.          |
                   |                       |                      |                       |
                   =========================                      =========================
                       |             |                                        |
                       |             |                                        |
                      \|/           \|/                                      \|/
============================  ============================   ============================
| config_item_node object  |  | config_item_node object  |   | config_item_node object  |
|                          |  |                          |   |                          |
| Only containing the      |  | Only containing the      |   | Only containing the      |
| stringlist object        |  | stringlist object        |   | stringlist object        |
| directly parsed from the |  | directly parsed from the |   | directly parsed from the |
| file.                    |  | file.                    |   | file.                    |
|--------------------------|  |--------------------------|   |--------------------------|
| ARG1 ARG2 ARG3           |  | VERBOSE                  |   | DEBUG                    |
============================  ============================   ============================


The example illustrated above would correspond to the following config
file (invariant under line-permutations):

KEY1   ARG1 ARG2 ARG3
KEY1   VERBOSE
KEY2   DEBUG


Example config file(2):

OUTFILE   filename
INPUT     filename
OPTIONS   store
OPTIONS   verbose
OPTIONS   optimize cache=1

In this case the whole config object will contain three items,
corresponding to the keywords OUTFILE, INPUT and OPTIONS. The two
first will again only contain one node each, whereas the OPTIONS item
will contain three nodes, corresponding to the three times the keyword
"OPTIONS" appear in the config file.
*/
struct config_parser_struct {
    hash_type *schema_items;
    /** Can print a (warning) message when a keyword is encountered. */
    hash_type *messages;
};

int config_get_schema_size(const config_parser_type *config) {
    return hash_get_size(config->schema_items);
}

/**
  The last argument (config_file) is only used for printing
  informative error messages, and can be NULL. The config_cwd is
  essential if we are looking up a filename, otherwise it can be NULL.

  Returns a string with an error description, or NULL if the supplied
  arguments were OK. The string is allocated here, but is assumed that
  calling scope will free it.
*/
static config_content_node_type *config_content_item_set_arg__(
    subst_list_type *define_list, std::vector<std::string> &parse_errors,
    config_content_item_type *item, stringlist_type *token_list,
    const config_path_elm_type *path_elm, const char *config_file) {

    config_content_node_type *new_node = NULL;
    int argc = stringlist_get_size(token_list) - 1;

    if (argc == 1 &&
        (strcmp(stringlist_iget(token_list, 1), CLEAR_STRING) == 0)) {
        config_content_item_clear(item);
    } else {
        const config_schema_item_type *schema_item =
            config_content_item_get_schema(item);

        /* Filtering based on DEFINE statements */
        if (subst_list_get_size(define_list) > 0 &&
            config_schema_item_substitutions_enabled(schema_item)) {
            char *parsing_line =
                stringlist_alloc_joined_string(token_list, " ");
            char *config_file_path =
                config_path_elm_alloc_abspath(path_elm, config_file);
            std::string context =
                fmt::format("parsing config file `{}` line: `{}`",
                            config_file_path, parsing_line);
            int iarg;
            for (iarg = 0; iarg < argc; iarg++) {

                char *filtered_copy = subst_list_alloc_filtered_string(
                    define_list, stringlist_iget(token_list, iarg + 1),
                    context.c_str(), 1000);
                stringlist_iset_owned_ref(token_list, iarg + 1, filtered_copy);
            }
        }

        /* Filtering based on environment variables */
        if (config_schema_item_expand_envvar(schema_item)) {
            int iarg;
            for (iarg = 0; iarg < argc; iarg++) {
                int env_offset = 0;
                while (true) {
                    char *env_var = res_env_isscanf_alloc_envvar(
                        stringlist_iget(token_list, iarg + 1), env_offset);
                    if (env_var == NULL)
                        break;

                    {
                        const char *env_value = getenv(&env_var[1]);
                        if (env_value != NULL) {
                            char *new_value = util_string_replace_alloc(
                                stringlist_iget(token_list, iarg + 1), env_var,
                                env_value);
                            stringlist_iset_owned_ref(token_list, iarg + 1,
                                                      new_value);
                        } else {
                            env_offset += 1;
                            logger->warning(
                                "Environment variable: {} is not defined",
                                env_var);
                        }
                    }

                    free(env_var);
                }
            }
        }

        {
            if (config_schema_item_validate_set(schema_item, token_list,
                                                config_file, path_elm,
                                                parse_errors)) {
                new_node = config_content_item_alloc_node(
                    item, config_content_item_get_path_elm(item));
                config_content_node_set(new_node, token_list);
            }
        }
    }
    return new_node;
}

config_parser_type *config_alloc() {
    config_parser_type *config =
        (config_parser_type *)util_malloc(sizeof *config);
    config->schema_items = hash_alloc();
    config->messages = hash_alloc();
    return config;
}

void config_free(config_parser_type *config) {

    hash_free(config->schema_items);
    hash_free(config->messages);

    free(config);
}

static void config_insert_schema_item(config_parser_type *config,
                                      const char *kw,
                                      const config_schema_item_type *item,
                                      bool ref) {
    if (ref)
        hash_insert_ref(config->schema_items, kw, item);
    else
        hash_insert_hash_owned_ref(config->schema_items, kw, item,
                                   config_schema_item_free__);
}

/**
   This function allocates a simple item with all values
   defaulted. The item is added to the config object, and a pointer is
   returned to the calling scope. If you want to change the properties
   of the item you can do that with config_schema_item_set_xxxx() functions
   from the calling scope.
*/
config_schema_item_type *config_add_schema_item(config_parser_type *config,
                                                const char *kw, bool required) {

    config_schema_item_type *item = config_schema_item_alloc(kw, required);
    config_insert_schema_item(config, kw, item, false);
    return item;
}

/**
  This is a minor wrapper for adding an item with the properties.

    1. It has argc_minmax = {1,1}

   The value can than be extracted with config_get_value() and
   config_get_value_as_xxxx functions.
*/
config_schema_item_type *config_add_key_value(config_parser_type *config,
                                              const char *key, bool required,
                                              config_item_types item_type) {
    config_schema_item_type *item =
        config_add_schema_item(config, key, required);
    config_schema_item_set_argc_minmax(item, 1, 1);
    config_schema_item_iset_type(item, 0, item_type);
    return item;
}

bool config_has_schema_item(const config_parser_type *config, const char *kw) {
    return hash_has_key(config->schema_items, kw);
}

config_schema_item_type *
config_get_schema_item(const config_parser_type *config, const char *kw) {
    return (config_schema_item_type *)hash_get(config->schema_items, kw);
}

/**
  Due to the possibility of aliases we must go through the canonical
  keyword which is internalized in the schema_item.
*/
static void config_validate_content_item(const config_parser_type *config,
                                         config_content_type *content,
                                         const config_content_item_type *item) {
    const config_schema_item_type *schema_item =
        config_content_item_get_schema(item);
    const char *schema_kw = config_schema_item_get_kw(schema_item);

    {
        int i;
        for (i = 0; i < config_schema_item_num_required_children(schema_item);
             i++) {
            const char *required_child =
                config_schema_item_iget_required_child(schema_item, i);
            if (!config_content_has_item(content, required_child)) {
                std::string error_message =
                    util_alloc_sprintf("When:%s is set - you also must set:%s.",
                                       schema_kw, required_child);
                content->parse_errors.push_back(error_message);
            }
        }

        if (config_schema_item_has_required_children_value(schema_item)) {
            int inode;
            for (inode = 0; inode < config_content_item_get_size(item);
                 inode++) {
                config_content_node_type *node =
                    config_content_item_iget_node(item, inode);
                const stringlist_type *values =
                    config_content_node_get_stringlist(node);
                int is;

                for (is = 0; is < stringlist_get_size(values); is++) {
                    const char *value = stringlist_iget(values, is);
                    stringlist_type *required_children =
                        config_schema_item_get_required_children_value(
                            schema_item, value);

                    if (required_children != NULL) {
                        int ic;
                        for (ic = 0;
                             ic < stringlist_get_size(required_children);
                             ic++) {
                            const char *req_child =
                                stringlist_iget(required_children, ic);
                            if (!config_content_has_item(content, req_child)) {
                                std::string error_message = util_alloc_sprintf(
                                    "When:%s is set to:%s - you also must "
                                    "set:%s.",
                                    schema_kw, value, req_child);
                                content->parse_errors.push_back(error_message);
                            }
                        }
                    }
                }
            }
        }
    }
}

void config_validate(config_parser_type *config, config_content_type *content) {
    int size = hash_get_size(config->schema_items);
    char **key_list = hash_alloc_keylist(config->schema_items);
    int ikey;
    for (ikey = 0; ikey < size; ikey++) {
        const config_schema_item_type *schema_item =
            config_get_schema_item(config, key_list[ikey]);
        const char *content_key = config_schema_item_get_kw(schema_item);
        if (config_content_has_item(content, content_key)) {
            const config_content_item_type *item =
                config_content_get_item(content, content_key);
            config_validate_content_item(config, content, item);
        } else {
            if (config_schema_item_required(schema_item)) {
                std::string error_message = util_alloc_sprintf(
                    "Item:%s must be set - parsing:%s", content_key,
                    config_content_get_config_file(content, true));
                content->parse_errors.push_back(error_message);
            }
        }
    }
    util_free_stringlist(key_list, size);
}

static void assert_no_circular_includes(config_content_type *config_content,
                                        const char *config_filename) {
    char *abs_filename = (char *)util_alloc_realpath(config_filename);
    if (!config_content_add_file(config_content, abs_filename))
        util_exit("%s: file (%s) already parsed - circular include?", __func__,
                  abs_filename);

    free(abs_filename);
}

static void alloc_config_filename_components(const char *config_filename,
                                             char **config_path,
                                             char **config_file) {
    char *config_base;
    char *config_ext;
    util_alloc_file_components(config_filename, config_path, &config_base,
                               &config_ext);

    *config_file = util_alloc_filename(NULL, config_base, config_ext);

    free(config_base);
    free(config_ext);
}

static config_path_elm_type *
config_relocate(const char *config_path, config_content_type *config_content,
                path_stack_type *path_stack) {
    config_path_elm_type *current_path_elm =
        config_content_add_path_elm(config_content, config_path);
    path_stack_push_cwd(path_stack);

    if (config_path)
        util_chdir(config_path);

    return current_path_elm;
}

bool config_parser_add_key_values(
    config_parser_type *config, config_content_type *content, const char *kw,
    stringlist_type *values, const config_path_elm_type *current_path_elm,
    const char *config_filename, config_schema_unrecognized_enum unrecognized) {
    if (!config_has_schema_item(config, kw)) {

        if (unrecognized == CONFIG_UNRECOGNIZED_IGNORE)
            return false;

        if (unrecognized == CONFIG_UNRECOGNIZED_WARN) {
            logger->warning(
                "** Warning keyword: {} not recognized when parsing: {} ---",
                kw, config_filename);
            return false;
        }

        if (unrecognized == CONFIG_UNRECOGNIZED_ERROR) {
            std::string error_message =
                util_alloc_sprintf("Keyword:%s is not recognized", kw);
            content->parse_errors.push_back(error_message);
            return false;
        }

        /*
      We allow unrecognized keywords - they are automatically added to the
       current parser class.
    */
        if (unrecognized == CONFIG_UNRECOGNIZED_ADD) {
            config_schema_item_type *item =
                config_add_schema_item(config, kw, false);
            config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);
        }
    }
    config_schema_item_type *schema_item = config_get_schema_item(config, kw);

    if (hash_has_key(config->messages, kw))
        printf("%s \n", (const char *)hash_get(config->messages, kw));

    if (!config_content_has_item(content, kw))
        config_content_add_item(content, schema_item, current_path_elm);

    config_content_item_type *content_item = config_content_get_item(
        content, config_schema_item_get_kw(schema_item));

    config_content_node_type *new_node = config_content_item_set_arg__(
        config_content_get_define_list(content), content->parse_errors,
        content_item, values, current_path_elm, config_filename);

    if (new_node)
        config_content_add_node(content, new_node);

    return new_node != NULL;
}

static std::optional<std::string> char_to_optstr(std::unique_ptr<char> p) {
    return p ? std::optional<std::string>(p.get()) : std::nullopt;
}

/**
   This function parses the config file 'filename', and updated the
   internal state of the config object as parsing proceeds. If
   comment_string != NULL everything following 'comment_string' on a
   line is discarded.

   include_kw is a string identifier for an include functionality, if
   an include is encountered, the included file is parsed immediately
   (through a recursive call to config_parse__). if include_kw == NULL,
   include files are not supported.

   Observe that the use of include and relative paths is
   quite tricky. The following is currently implemented:

    1. The front_end function will split the path to the config file
       in a path_name component and a file component.

    2. Recursive calls to config_parse__() will keep control of the
       parsers notion of cwd (note that the real OS'wise cwd never
       changes), and every item is tagged with the config_cwd
       currently active.

    3. When an item has been entered with type CONFIG_FILE /
       CONFIG_DIRECTORY / CONFIG_EXECUTABLE - the item is updated to
       reflect to be relative (iff it is relative in the first place)
       to the path of the root config file.

   These are not strict rules - it is possible to get other things to
   work as well, but the problem is that it very quickly becomes
   dependent on 'arbitrariness' in the parsing configuration.

   validate: whether we should validate when complete, that should
             typically only be done at the last parsing.


   define_kw: This a string which can serve as a "#define" for the
   parsing. The define_kw keyword should have two arguments - a key
   and a value. If the define_kw is present all __subsequent__
   occurences of 'key' are replaced with 'value'.  alloc_new_key
   is an optinal function (can be NULL) which is used to alloc a new
   key, i.e. add leading and trailing 'magic' characters.


   Example:
   --------

   char * add_angular_brackets(const char * key) {
       char * new_key = (char*)util_alloc_sprintf("<%s>" , key);
   }



   config_parse(... , "myDEF" , add_angular_brackets , ...)


   Config file:
   -------------
   myDEF   Name         BJARNE
   myDEF   pet        Dog
   ...
   ...
   PERSON  <Name> 28 <pet>
   ...
   ------------

   After parsing we will have an entry: "NAME" , "Bjarne" , "28" , "Dog".

   The         key-value pairs internalized during the config parsing are NOT
   returned to the calling scope in any way.
*/
static void
config_parse__(config_parser_type *config, config_content_type *content,
               path_stack_type *path_stack, const char *config_filename,
               const char *comment_string, const char *include_kw,
               const char *define_kw,
               config_schema_unrecognized_enum unrecognized, bool validate) {
    assert_no_circular_includes(content, config_filename);

    // Relocate
    char *config_path;
    char *config_file;
    alloc_config_filename_components(config_filename, &config_path,
                                     &config_file);

    config_path_elm_type *current_path_elm =
        config_relocate(config_path, content, path_stack);
    free(config_path);

    // Setup config parsing
    const char *comment_end = comment_string ? "\n" : NULL;
    basic_parser_type *parser = basic_parser_alloc(" \t", "\"", NULL, NULL,
                                                   comment_string, comment_end);

    FILE *stream = util_fopen(config_file, "r");
    bool at_eof = false;

    while (!at_eof) {
        auto line_buffer = char_to_optstr(
            std::unique_ptr<char>(util_fscanf_alloc_line(stream, &at_eof)));
        if (!line_buffer.has_value())
            continue;

        const std::unique_ptr<stringlist_type, void (*)(stringlist_type *)>
            token_list(basic_parser_tokenize_buffer(
                           parser, line_buffer.value().c_str(), true),
                       stringlist_free);
        const int active_tokens = stringlist_get_size(token_list.get());

        if (active_tokens > 0) {
            const char *kw = stringlist_iget(token_list.get(), 0);

            // Include config file
            if (include_kw && (strcmp(include_kw, kw) == 0)) {
                if (active_tokens != 2) {
                    content->parse_errors.push_back(fmt::format(
                        "Keyword:{} must have exactly one argument.",
                        include_kw));
                    return;
                }

                const char *include_file = stringlist_iget(token_list.get(), 1);

                if (!fs::exists(include_file)) {
                    content->parse_errors.push_back(fmt::format(
                        "{} file:{} not found", include_kw, include_file));
                } else {
                    config_parse__(config, content, path_stack, include_file,
                                   comment_string, include_kw, define_kw,
                                   unrecognized, false);
                }
            }

            // Add define
            else if (define_kw && (strcmp(define_kw, kw) == 0)) {
                if (active_tokens < 3) {
                    content->parse_errors.push_back(fmt::format(
                        "Keyword:{} must have two or more arguments.",
                        define_kw));
                    return;
                }

                const std::string key = stringlist_iget(token_list.get(), 1);
                const int argc = stringlist_get_size(token_list.get());
                for (int iarg = 2; iarg < argc; iarg++) {
                    int env_offset = 0;
                    while (true) {
                        auto env_var = char_to_optstr(
                            std::unique_ptr<char>(res_env_isscanf_alloc_envvar(
                                stringlist_iget(token_list.get(), iarg),
                                env_offset)));
                        if (!env_var.has_value())
                            break;

                        const char *env_value =
                            getenv(env_var.value().substr(1).c_str());
                        if (env_value != nullptr) {
                            stringlist_iset_owned_ref(
                                token_list.get(), iarg,
                                util_string_replace_alloc(
                                    stringlist_iget(token_list.get(), iarg),
                                    env_var.value().c_str(), env_value));
                        } else {
                            env_offset += 1;
                            logger->warning(
                                "Environment variable: {} is not defined",
                                env_var.value());
                        }
                    }
                }
                config_content_add_define(
                    content, key.c_str(),
                    std::unique_ptr<char>(
                        stringlist_alloc_joined_substring(token_list.get(), 2,
                                                          active_tokens, " "))
                        .get());
            }

            // Add keyword
            else
                config_parser_add_key_values(config, content, kw,
                                             token_list.get(), current_path_elm,
                                             config_file, unrecognized);
        }
    }

    fclose(stream);
    basic_parser_free(parser);

    if (validate)
        config_validate(config, content);

    free(config_file);
    path_stack_pop(path_stack);
    config_content_pop_path_stack(content);
}

config_content_type *
config_parse(config_parser_type *config, const char *filename,
             const char *comment_string, const char *include_kw,
             const char *define_kw, const hash_type *pre_defined_kw_map,
             config_schema_unrecognized_enum unrecognized_behaviour,
             bool validate) {

    config_content_type *content = config_content_alloc(filename);

    if (pre_defined_kw_map != NULL) {
        hash_iter_type *keys = hash_iter_alloc(pre_defined_kw_map);

        while (!hash_iter_is_complete(keys)) {
            const char *key = hash_iter_get_next_key(keys);
            const char *value = (const char *)hash_get(pre_defined_kw_map, key);
            config_content_add_define(content, key, value);
        }

        hash_iter_free(keys);
    }

    bool file_readable_check_succeeded = true;
    try {
        std::ifstream file_handler;
        file_handler.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        file_handler.open(filename);
    } catch (std::ios_base::failure &err) {
        file_readable_check_succeeded = false;
        auto error_message = fmt::format(
            "could not open file `{}` for parsing - {}", filename, err.what());
        content->parse_errors.push_back(error_message);
    }

    if (file_readable_check_succeeded) {
        path_stack_type *path_stack = path_stack_alloc();
        config_parse__(config, content, path_stack, filename, comment_string,
                       include_kw, define_kw, unrecognized_behaviour, validate);
        path_stack_free(path_stack);
    }

    if (content->parse_errors.size() == 0)
        config_content_set_valid(content);

    return content;
}

/**
   This function adds an alias to an existing item; so that the
   value+++ of an item can be referred to by two different names.
*/
void config_add_alias(config_parser_type *config, const char *src,
                      const char *alias) {
    if (config_has_schema_item(config, src)) {
        config_schema_item_type *item = config_get_schema_item(config, src);
        config_insert_schema_item(config, alias, item, true);
    } else
        util_abort("%s: item:%s not recognized \n", __func__, src);
}

void config_install_message(config_parser_type *config, const char *kw,
                            const char *message) {
    hash_insert_hash_owned_ref(config->messages, kw,
                               util_alloc_string_copy(message), free);
}
