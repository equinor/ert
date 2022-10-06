#include <filesystem>

#include <ert/config/config_content.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/enkf/subst_config.hpp>

namespace fs = std::filesystem;

struct subst_config_struct {

    subst_func_pool_type *subst_func_pool;
    subst_list_type *subst_list;
};

static void subst_config_init_default(subst_config_type *subst_config);

static void
subst_config_install_config_directory(subst_config_type *subst_config,
                                      const char *user_config_file);

static void subst_config_init_load(subst_config_type *subst_config,
                                   const config_content_type *content);

static subst_config_type *subst_config_alloc_empty() {
    subst_config_type *subst_config =
        (subst_config_type *)util_malloc(sizeof *subst_config);

    subst_config->subst_func_pool = NULL;
    subst_config->subst_list = NULL;

    return subst_config;
}

static subst_config_type *subst_config_alloc_default() {
    subst_config_type *subst_config = subst_config_alloc_empty();

    subst_config->subst_func_pool = subst_func_pool_alloc();
    subst_config->subst_list = subst_list_alloc(subst_config->subst_func_pool);

    subst_config_init_default(subst_config);

    return subst_config;
}

subst_config_type *subst_config_alloc(const config_content_type *user_config) {
    subst_config_type *subst_config = subst_config_alloc_default();

    if (user_config)
        subst_config_init_load(subst_config, user_config);

    return subst_config;
}

subst_config_type *subst_config_alloc_full(const subst_list_type *define_list) {
    subst_config_type *subst_config = subst_config_alloc_default();

    // copy list of substitution keywords
    for (int i = 0; i < subst_list_get_size(define_list); i++) {
        const char *key = subst_list_iget_key(define_list, i);
        const char *value = subst_list_iget_value(define_list, i);
        const char *docstring = subst_list_get_doc_string(define_list, key);
        subst_list_append_copy(subst_config->subst_list, key, value, docstring);
    }

    return subst_config;
}

void subst_config_free(subst_config_type *subst_config) {
    if (!subst_config)
        return;

    subst_func_pool_free(subst_config->subst_func_pool);
    subst_list_free(subst_config->subst_list);

    free(subst_config);
}

subst_list_type *subst_config_get_subst_list(subst_config_type *subst_type) {
    return subst_type->subst_list;
}

void subst_config_add_internal_subst_kw(subst_config_type *subst_config,
                                        const char *key, const char *value,
                                        const char *help_text) {
    char *tagged_key = util_alloc_sprintf(INTERNAL_DATA_KW_TAG_FORMAT, key);
    subst_list_append_copy(subst_config_get_subst_list(subst_config),
                           tagged_key, value, help_text);
    free(tagged_key);
}

void subst_config_add_subst_kw(subst_config_type *subst_config, const char *key,
                               const char *value) {
    subst_list_append_copy(subst_config->subst_list, key, value,
                           "Supplied by the user in the configuration file.");
}

static void subst_config_install_num_cpu(subst_config_type *subst_config,
                                         int num_cpu) {
    char *num_cpu_string = util_alloc_sprintf("%d", num_cpu);
    subst_config_add_internal_subst_kw(
        subst_config, "NUM_CPU", num_cpu_string,
        "The number of CPU used for one forward model.");
    free(num_cpu_string);
}

static void subst_config_init_default(subst_config_type *subst_config) {
    /* Here we add the functions which should be available for string substitution operations. */

    subst_func_pool_add_func(subst_config->subst_func_pool, "EXP", "exp",
                             subst_func_exp, false, 1, 1, NULL);
    subst_func_pool_add_func(subst_config->subst_func_pool, "LOG", "log",
                             subst_func_log, false, 1, 1, NULL);
    subst_func_pool_add_func(subst_config->subst_func_pool, "POW10",
                             "Calculates 10^x", subst_func_pow10, false, 1, 1,
                             NULL);
    subst_func_pool_add_func(subst_config->subst_func_pool, "ADD",
                             "Adds arguments", subst_func_add, true, 1, 0,
                             NULL);
    subst_func_pool_add_func(subst_config->subst_func_pool, "MUL",
                             "Multiplies arguments", subst_func_mul, true, 1, 0,
                             NULL);

    /*
     Allocating the parent subst_list instance. This will (should ...)
     be the top level subst instance for all substitions in the ert
     program.

     All the functions available or only installed in this
     subst_list.

     The key->value replacements installed in this instance are
     key,value pairs which are:

      o Common to all ensemble members.

      o Constant in time.
    */

    /* Installing the functions. */
    subst_list_insert_func(subst_config->subst_list, "EXP", "__EXP__");
    subst_list_insert_func(subst_config->subst_list, "LOG", "__LOG__");
    subst_list_insert_func(subst_config->subst_list, "POW10", "__POW10__");
    subst_list_insert_func(subst_config->subst_list, "ADD", "__ADD__");
    subst_list_insert_func(subst_config->subst_list, "MUL", "__MUL__");

    // Installing the based (key,value) pairs which are common to all ensemble
    // members, and independent of time.

    int mday, year, month;
    util_set_datetime_values_utc(time(NULL), NULL, NULL, NULL, &mday, &month,
                                 &year);
    char *date_string = util_alloc_sprintf("%4d-%02d-%02d", year, month, mday);
    subst_config_add_internal_subst_kw(subst_config, "DATE", date_string,
                                       "The current date.");
    free(date_string);

    subst_config_install_num_cpu(subst_config, 1);
}

static void
subst_config_install_config_directory(subst_config_type *subst_config,
                                      const char *config_dir) {
    subst_config_add_internal_subst_kw(
        subst_config, "CWD", config_dir,
        "The current working directory we are running from - the location of "
        "the config file.");
    subst_config_add_internal_subst_kw(
        subst_config, "CONFIG_PATH", config_dir,
        "The current working directory we are running from - the location of "
        "the config file.");
}

static void subst_config_install_data_kw(subst_config_type *subst_config,
                                         hash_type *config_data_kw) {
    // Installing the DATA_KW keywords supplied by the user - these are at the
    // very top level, so they can reuse everything defined later.
    if (config_data_kw) {
        hash_iter_type *iter = hash_iter_alloc(config_data_kw);
        const char *key = hash_iter_get_next_key(iter);
        while (key != NULL) {
            subst_config_add_subst_kw(
                subst_config, key, (const char *)hash_get(config_data_kw, key));
            key = hash_iter_get_next_key(iter);
        }
        hash_iter_free(iter);
    }
}

static void subst_config_init_load(subst_config_type *subst_config,
                                   const config_content_type *content) {

    if (config_content_has_item(content, CONFIG_DIRECTORY_KEY)) {
        const char *work_dir =
            config_content_get_value_as_abspath(content, CONFIG_DIRECTORY_KEY);
        subst_config_install_config_directory(subst_config, work_dir);
    }

    const subst_list_type *define_list =
        config_content_get_const_define_list(content);
    for (int i = 0; i < subst_list_get_size(define_list); i++) {
        const char *key = subst_list_iget_key(define_list, i);
        const char *value = subst_list_iget_value(define_list, i);
        subst_config_add_subst_kw(subst_config, key, value);
    }

    if (config_content_has_item(content, DATA_KW_KEY)) {
        config_content_item_type *data_item =
            config_content_get_item(content, DATA_KW_KEY);
        hash_type *data_kw = config_content_item_alloc_hash(data_item, true);
        subst_config_install_data_kw(subst_config, data_kw);
        hash_free(data_kw);
    }

    const char *runpath_file =
        config_content_has_item(content, RUNPATH_FILE_KEY)
            ? config_content_get_value_as_abspath(content, RUNPATH_FILE_KEY)
            : util_alloc_filename(config_content_get_config_path(content),
                                  ".ert_runpath_list", NULL);
    subst_config_add_internal_subst_kw(
        subst_config, "RUNPATH_FILE", runpath_file,
        "The name of a file with a list of run directories.");

    if (config_content_has_item(content, DATA_FILE_KEY)) {
        const char *data_file =
            config_content_get_value_as_abspath(content, DATA_FILE_KEY);

        if (!fs::exists(data_file))
            util_abort("%s: Could not find ECLIPSE data file: %s\n", __func__,
                       data_file ? data_file : "NULL");

        int num_cpu = ecl_util_get_num_cpu(data_file);
        subst_config_install_num_cpu(subst_config, num_cpu);
    }

    // The NUM_CPU keyword in the user's config overrides the one set by the DATA_FILE above
    if (config_content_has_item(content, NUM_CPU_KEY)) {
        int num_cpu = config_content_get_value_as_int(content, NUM_CPU_KEY);
        subst_config_install_num_cpu(subst_config, num_cpu);
    }
}
