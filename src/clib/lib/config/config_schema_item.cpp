#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include <stdlib.h>
#include <string.h>

#include <fmt/format.h>

#include <ert/res_util/res_env.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/parser.hpp>

#include <ert/config/config_schema_item.hpp>

namespace fs = std::filesystem;

typedef struct validate_struct validate_type;

/**
   This is a 'support-struct' holding various pieces of information
   needed during the validation process. Observe the following about
   validation:

    1. It is atomic, in the sense that if you try to an item like this:

         KW  ARG1 ARG2 ARG3

       where ARG1 and ARG2 are valid, whereas there is something wrong
       with ARG3, NOTHING IS SET.

    2. Validation is a two-step process, the first step is run when an
       item is parsed. This includes checking:

        o The number of argument.
        o That the arguments have the right type.
        o That the values match the selection set.

       The second validation step is done when the pasing is complete,
       in this pass we check dependencies - i.e. required_children and
       required_children_on_value.


   Observe that nothing has-to be set in this struct. There are some dependencies:

    1. Only _one_ of common_selection_set and indexed_selection_set
       can be set.

    2. If setting indexed_selection_set or type_map, you MUST set
       argc_max first.
*/
struct validate_struct {
    std::set<std::string> common_selection_set; /** A selection set which will
                                                  apply uniformly to all the
                                                  arguments. */
    std::vector<std::set<std::string>> indexed_selection_set;

    /** The minimum number of arguments: -1 means no lower limit. */
    int argc_min;
    /** The maximum number of arguments: -1 means no upper limit. */
    int argc_max;
    /** A list of types for the items. Set along with argc_minmax(); */
    int_vector_type *type_map;
    /** A list of item's which must also be set (if this item is set). (can be
     * NULL) */
    stringlist_type *required_children;
    /** A list of item's which must also be set - depending on the value of
     * this item. (can be NULL) (This one is complex). */
    hash_type *required_children_value;
};

struct config_schema_item_struct {
    /** The kw which identifies this item */
    char *kw;

    bool required_set;
    /** A list of item's which must also be set (if this item is set). (can be
     * NULL) */
    stringlist_type *required_children;
    /** A list of item's which must also be set - depending on the value of
     * this item. (can be NULL) */
    hash_type *required_children_value;
    /** Information need during validation. */
    validate_type *validate;
    /** Should environment variables like $HOME be expanded?*/
    bool expand_envvar;
    bool deprecated;
    char *deprecate_msg;
    bool do_substitutions;
};

static void validate_set_default_type(validate_type *validate,
                                      config_item_types item_type) {
    int_vector_set_default(validate->type_map, item_type);
}

static validate_type *validate_alloc() {
    validate_type *validate = new validate_type();
    validate->argc_min = CONFIG_DEFAULT_ARG_MIN;
    validate->argc_max = CONFIG_DEFAULT_ARG_MAX;
    validate->required_children = NULL;
    validate->required_children_value = NULL;
    validate->type_map = int_vector_alloc(0, 0);
    validate_set_default_type(validate, CONFIG_STRING);
    return validate;
}

static void validate_free(validate_type *validate) {
    int_vector_free(validate->type_map);
    if (validate->required_children != NULL)
        stringlist_free(validate->required_children);
    if (validate->required_children_value != NULL)
        hash_free(validate->required_children_value);
    delete validate;
}

static void validate_iset_type(validate_type *validate, int index,
                               config_item_types type) {
    int_vector_iset(validate->type_map, index, type);
}

static config_item_types validate_iget_type(const validate_type *validate,
                                            int index) {
    return (config_item_types)int_vector_safe_iget(validate->type_map, index);
}

static void validate_set_argc_minmax(validate_type *validate, int argc_min,
                                     int argc_max) {
    if (validate->argc_min != CONFIG_DEFAULT_ARG_MIN)
        util_abort("%s: sorry - current implementation does not allow repeated "
                   "calls to: %s \n",
                   __func__, __func__);

    if (argc_min == CONFIG_DEFAULT_ARG_MIN)
        argc_min = 0;

    validate->argc_min = argc_min;
    validate->argc_max = argc_max;

    if ((argc_max != CONFIG_DEFAULT_ARG_MAX) && (argc_max < argc_min))
        util_abort("%s invalid arg min/max values. argc_min:%d  argc_max:%d \n",
                   __func__, argc_min, argc_max);
}

static void validate_set_common_selection_set(validate_type *validate,
                                              const stringlist_type *argv) {
    validate->common_selection_set.clear();
    for (int i = 0; i < stringlist_get_size(argv); i++)
        validate->common_selection_set.insert(stringlist_iget(argv, i));
}

static std::set<std::string> *
validate_iget_selection_set(validate_type *validate, int index) {
    if (index >= validate->indexed_selection_set.size())
        return nullptr;

    return &validate->indexed_selection_set[index];
}

static void validate_add_indexed_alternative(validate_type *validate, int index,
                                             const char *value) {
    if (index >= validate->indexed_selection_set.size())
        validate->indexed_selection_set.resize(index + 1);

    auto &set = validate->indexed_selection_set[index];
    set.insert(value);
}

static void validate_set_indexed_selection_set(validate_type *validate,
                                               int index,
                                               const stringlist_type *argv) {
    if (index >= validate->argc_min)
        util_abort("%s: When not not setting argc_max selection set can only "
                   "be applied to indices up to argc_min\n",
                   __func__);

    if (index >= validate->indexed_selection_set.size())
        validate->indexed_selection_set.resize(index + 1);

    auto &set = validate->indexed_selection_set[index];
    for (int i = 0; i < stringlist_get_size(argv); i++)
        set.insert(stringlist_iget(argv, i));
}

void config_schema_item_assure_type(const config_schema_item_type *item,
                                    int index, int type_mask) {
    bool OK = false;

    if (int_vector_safe_iget(item->validate->type_map, index) & type_mask)
        OK = true;

    if (!OK)
        util_abort("%s: failed - wrong installed type \n", __func__);
}

config_schema_item_type *config_schema_item_alloc(const char *kw,
                                                  bool required) {
    config_schema_item_type *item =
        (config_schema_item_type *)util_malloc(sizeof *item);
    item->kw = util_alloc_string_copy(kw);

    item->required_set = required;
    item->deprecated = false;
    item->deprecate_msg = NULL;
    item->required_children = NULL;
    item->required_children_value = NULL;
    item->expand_envvar = true; // Default is to expand $VAR expressions;
                                // can be turned off with
                                // config_schema_item_set_envvar_expansion(
                                //     item,
                                //     false);
    item->validate = validate_alloc();
    item->do_substitutions = true;
    return item;
}

bool config_schema_item_valid_string(config_item_types value_type,
                                     const char *value, bool runtime) {
    switch (value_type) {
    case (CONFIG_ISODATE):
        return util_sscanf_isodate(value, NULL);
        break;
    case (CONFIG_INT):
        return util_sscanf_int(value, NULL);
        break;
    case (CONFIG_FLOAT):
        return util_sscanf_double(value, NULL);
        break;
    case (CONFIG_BOOL):
        return util_sscanf_bool(value, NULL);
        break;
    case (CONFIG_BYTESIZE):
        return util_sscanf_bytesize(value, NULL);
        break;
    case (CONFIG_RUNTIME_INT):
        if (runtime)
            return util_sscanf_int(value, NULL);
        else
            return true;
        break;
    case (CONFIG_RUNTIME_FILE):
        if (runtime)
            return fs::exists(value);
        else
            return true;
        break;
    default:
        return true;
    }
}

static char *alloc_relocated__(const config_path_elm_type *path_elm,
                               const char *value) {
    if (util_is_abs_path(value))
        return util_alloc_string_copy(value);

    return util_alloc_filename(config_path_elm_get_abspath(path_elm), value,
                               NULL);
}

bool config_schema_item_validate_set(const config_schema_item_type *item,
                                     stringlist_type *token_list,
                                     const char *config_file,
                                     const config_path_elm_type *path_elm,
                                     std::vector<std::string> &error_list) {
    bool OK = true;
    int argc = stringlist_get_size(token_list) - 1;
    if (item->validate->argc_min >= 0) {
        if (argc < item->validate->argc_min) {
            OK = false;
            char *error_message;
            if (config_file != NULL)
                error_message = util_alloc_sprintf(
                    "Error when parsing config_file:\"%s\" Keyword:%s must "
                    "have at least %d arguments.",
                    config_file, item->kw, item->validate->argc_min);
            else
                error_message = util_alloc_sprintf(
                    "Error:: Keyword:%s must have at least %d arguments.",
                    item->kw, item->validate->argc_min);

            error_list.push_back(std::string(error_message));
        }
    }

    if (item->validate->argc_max >= 0) {
        if (argc > item->validate->argc_max) {
            OK = false;
            {
                char *error_message;

                if (config_file != NULL)
                    error_message = util_alloc_sprintf(
                        "Error when parsing config_file:\"%s\" Keyword:%s must "
                        "have maximum %d arguments.",
                        config_file, item->kw, item->validate->argc_max);
                else
                    error_message = util_alloc_sprintf(
                        "Error:: Keyword:%s must have maximum %d arguments.",
                        item->kw, item->validate->argc_max);

                error_list.push_back(std::string(error_message));
            }
        }
    }

    /*
     * OK - now we have verified that the number of arguments is correct. Then
     * we start actually looking at the values.
     */
    if (OK) {
        /* Validating selection set - first common, then indexed */
        if (item->validate->common_selection_set.size()) {
            for (int iarg = 0; iarg < argc; iarg++) {
                if (!item->validate->common_selection_set.count(
                        stringlist_iget(token_list, iarg + 1))) {
                    std::string error_message = util_alloc_sprintf(
                        "%s: is not a valid value for: %s.",
                        stringlist_iget(token_list, iarg + 1), item->kw);
                    error_list.push_back(error_message);
                    OK = false;
                }
            }
        } else if (item->validate->indexed_selection_set.size()) {
            for (int iarg = 0; iarg < argc; iarg++) {
                if ((item->validate->argc_max > 0) ||
                    (iarg <
                     item->validate
                         ->argc_min)) { /* Without this test we might go out of range on the indexed selection set. */
                    const auto *selection_set =
                        validate_iget_selection_set(item->validate, iarg);
                    if (selection_set && selection_set->size()) {
                        if (!selection_set->count(
                                stringlist_iget(token_list, iarg + 1))) {
                            std::string error_message = util_alloc_sprintf(
                                "%s: is not a valid value for item %d of "
                                "\'%s\'.",
                                stringlist_iget(token_list, iarg + 1), iarg + 1,
                                item->kw);
                            error_list.push_back(error_message);
                            OK = false;
                        }
                    }
                }
            }
        }

        /*
         * Observe that the following code might rewrite the content of
         * argv for arguments referring to path locations.
         */

        /* Validate the TYPE of the various argumnents */
        {
            for (int iarg = 0; iarg < argc; iarg++) {
                const char *value = stringlist_iget(token_list, iarg + 1);
                switch (validate_iget_type(item->validate, iarg)) {
                case (CONFIG_STRING): /* This never fails ... */
                    break;
                case (CONFIG_RUNTIME_INT):
                    break;
                case (CONFIG_RUNTIME_FILE):
                    break;
                case (CONFIG_ISODATE):
                    if (!util_sscanf_isodate(value, NULL))
                        error_list.push_back(
                            util_alloc_sprintf("Failed to parse:%s as an ISO "
                                               "date: YYYY-MM-DD.",
                                               value));
                    break;
                case (CONFIG_INT):
                    if (!util_sscanf_int(value, NULL))
                        error_list.push_back(util_alloc_sprintf(
                            "Failed to parse:%s as an integer.", value));
                    break;
                case (CONFIG_FLOAT):
                    if (!util_sscanf_double(value, NULL)) {
                        error_list.push_back(
                            util_alloc_sprintf("Failed to parse:%s as a "
                                               "floating point number.",
                                               value));
                        OK = false;
                    }
                    break;
                case (CONFIG_PATH):
                    // As long as we do not reuqire the path to exist it is just a string.
                    break;
                case (CONFIG_EXISTING_PATH): {
                    char *path = config_path_elm_alloc_abspath(path_elm, value);
                    if (!util_entry_exists(path)) {
                        error_list.push_back(util_alloc_sprintf(
                            "Cannot find file or directory \"%s\" in path "
                            "\"%s\" ",
                            value, config_path_elm_get_abspath(path_elm)));
                        OK = false;
                    }
                    free(path);
                } break;
                case (CONFIG_EXECUTABLE): {
                    if (!util_is_abs_path(value)) {

                        char *relocated = alloc_relocated__(path_elm, value);
                        if (fs::exists(relocated)) {
                            if (util_is_executable(relocated))
                                stringlist_iset_copy(token_list, iarg,
                                                     relocated);
                            else
                                error_list.push_back(fmt::format(
                                    "File not executable:{}", value));
                            free(relocated);
                            break;
                        }

                        free(relocated);

                        /*
                         * res_env_alloc_PATH_executable aborts if some parts of the path is
                         * not an existing dir, so call it only when its an absolute path
                         */
                        char *path_exe = res_env_alloc_PATH_executable(value);
                        if (path_exe != NULL)
                            stringlist_iset_copy(token_list, iarg, path_exe);
                        else
                            error_list.push_back(util_alloc_sprintf(
                                "Executable:%s does not exist", value));

                        free(path_exe);
                    } else {
                        if (!util_is_executable(value))
                            error_list.push_back(util_alloc_sprintf(
                                "File not executable:%s ", value));
                    }
                } break;
                case (CONFIG_BOOL):
                    if (!util_sscanf_bool(value, NULL)) {
                        error_list.push_back(util_alloc_sprintf(
                            "Failed to parse:%s as a boolean.", value));
                        OK = false;
                    }
                    break;
                case (CONFIG_BYTESIZE):
                    if (!util_sscanf_bytesize(value, NULL)) {
                        error_list.push_back(util_alloc_sprintf(
                            "Failed to parse:\"%s\" as number of bytes.",
                            value));
                        OK = false;
                    }
                    break;
                default:
                    util_abort("%s: config_item_type:%d not recognized \n",
                               __func__,
                               validate_iget_type(item->validate, iarg));
                }
            }
        }
    }
    return OK;
}

void config_schema_item_free(config_schema_item_type *item) {
    free(item->kw);
    free(item->deprecate_msg);
    if (item->required_children != NULL)
        stringlist_free(item->required_children);
    if (item->required_children_value != NULL)
        hash_free(item->required_children_value);
    validate_free(item->validate);
    free(item);
}

void config_schema_item_free__(void *void_item) {
    auto item = static_cast<config_schema_item_type *>(void_item);
    config_schema_item_free(item);
}

void config_schema_item_set_required_children_on_value(
    config_schema_item_type *item, const char *value,
    stringlist_type *child_list) {
    if (item->required_children_value == NULL)
        item->required_children_value = hash_alloc();
    hash_insert_hash_owned_ref(item->required_children_value, value,
                               stringlist_alloc_deep_copy(child_list),
                               stringlist_free__);
}

/**
   This function is used to set the minimum and maximum number of
   arguments for an item. In addition you can pass in a pointer to an
   array of config_schema_item_types values which will be used for validation
   of the input. This vector must be argc_max elements long; it can be
   NULL.
*/
void config_schema_item_set_argc_minmax(config_schema_item_type *item,
                                        int argc_min, int argc_max) {

    validate_set_argc_minmax(item->validate, argc_min, argc_max);
}

void config_schema_item_iset_type(config_schema_item_type *item, int index,
                                  config_item_types type) {
    validate_iset_type(item->validate, index, type);
}

void config_schema_item_disable_substitutions(config_schema_item_type *item) {
    item->do_substitutions = false;
}

bool config_schema_item_substitutions_enabled(
    const config_schema_item_type *item) {
    return item->do_substitutions;
}

void config_schema_item_set_default_type(config_schema_item_type *item,
                                         config_item_types type) {
    validate_set_default_type(item->validate, type);
}

config_item_types
config_schema_item_iget_type(const config_schema_item_type *item, int index) {
    return validate_iget_type(item->validate, index);
}

void config_schema_item_set_envvar_expansion(config_schema_item_type *item,
                                             bool expand_envvar) {
    item->expand_envvar = expand_envvar;
}

void config_schema_item_set_common_selection_set(config_schema_item_type *item,
                                                 const stringlist_type *argv) {
    validate_set_common_selection_set(item->validate, argv);
}

void config_schema_item_set_indexed_selection_set(config_schema_item_type *item,
                                                  int index,
                                                  const stringlist_type *argv) {
    validate_set_indexed_selection_set(item->validate, index, argv);
}

void config_schema_item_add_indexed_alternative(config_schema_item_type *item,
                                                int index, const char *value) {
    validate_add_indexed_alternative(item->validate, index, value);
}

void config_schema_item_add_required_children(config_schema_item_type *item,
                                              const char *child_key) {
    if (item->required_children == NULL)
        item->required_children = stringlist_alloc_new();

    stringlist_append_copy(item->required_children, child_key);
}

int config_schema_item_num_required_children(
    const config_schema_item_type *item) {
    if (item->required_children == NULL)
        return 0;
    else
        return stringlist_get_size(item->required_children);
}

const char *
config_schema_item_iget_required_child(const config_schema_item_type *item,
                                       int index) {
    return stringlist_iget(item->required_children, index);
}

const char *config_schema_item_get_kw(const config_schema_item_type *item) {
    return item->kw;
}

bool config_schema_item_required(const config_schema_item_type *item) {
    return item->required_set;
}

bool config_schema_item_expand_envvar(const config_schema_item_type *item) {
    return item->expand_envvar;
}

void config_schema_item_get_argc(const config_schema_item_type *item,
                                 int *argc_min, int *argc_max) {
    *argc_min = item->validate->argc_min;
    *argc_max = item->validate->argc_max;
}

bool config_schema_item_has_required_children_value(
    const config_schema_item_type *item) {
    if (item->required_children_value == NULL)
        return false;
    else
        return true;
}

stringlist_type *config_schema_item_get_required_children_value(
    const config_schema_item_type *item, const char *value) {
    return (stringlist_type *)hash_safe_get(item->required_children_value,
                                            value);
}

bool config_schema_item_is_deprecated(const config_schema_item_type *item) {
    return item->deprecated;
}

const char *
config_schema_item_get_deprecate_msg(const config_schema_item_type *item) {
    return item->deprecate_msg;
}

void config_schema_item_set_deprecated(config_schema_item_type *item,
                                       const char *msg) {
    item->deprecated = true;
    item->deprecate_msg = util_realloc_string_copy(item->deprecate_msg, msg);
}
