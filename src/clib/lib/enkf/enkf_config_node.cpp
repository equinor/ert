#include <filesystem>

#include <stdio.h>
#include <stdlib.h>

#include <ert/ecl/ecl_grid.h>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/ext_param_config.hpp>
#include <ert/enkf/field_config.hpp>
#include <ert/enkf/gen_kw_config.hpp>
#include <ert/enkf/gen_obs.hpp>
#include <ert/enkf/surface_config.hpp>
#include <ert/res_util/path_fmt.hpp>
#include <ert/util/string_util.h>
#include <ert/util/stringlist.h>
#include <ert/util/vector.h>

namespace fs = std::filesystem;

struct enkf_config_node_struct {
    ert_impl_type impl_type;
    enkf_var_type var_type;
    bool vector_storage;
    /** Should the (parameter) node be initialized by loading results from the
     * Forward model? */
    bool forward_init;

    /** Should this node be internalized - observe that question of what to
     * internalize is MOSTLY handled at a higher level - without consulting
     * this variable. Can be NULL. */
    bool_vector_type *internalize;
    /** Keys of observations which observe this node. */
    stringlist_type *obs_keys;
    char *key;
    char *init_file_abs_path;
    /** Format used to create files for initialization. */
    path_fmt_type *init_file_fmt;
    /** Format used to load in file from forward model - one %d (if present) is
     * replaced with report_step. */
    path_fmt_type *enkf_infile_fmt;
    /** Name of file which is written by EnKF, and read by the forward model. */
    path_fmt_type *enkf_outfile_fmt;
    /** This points to the config object of the actual implementation. */
    void *data;

    /** Function pointer to ask the underlying config object of the size - i.e.
     * number of elements. */
    get_data_size_ftype *get_data_size;
    config_free_ftype *freef;
};

bool enkf_config_node_has_node(const enkf_config_node_type *node,
                               enkf_fs_type *fs, node_id_type node_id) {

    return enkf_fs_has_node(fs, node->key, node->var_type, node_id.report_step,
                            node_id.iens);
}

bool enkf_config_node_has_vector(const enkf_config_node_type *node,
                                 enkf_fs_type *fs, int iens) {
    bool has_vector = enkf_fs_has_vector(fs, node->key, node->var_type, iens);
    return has_vector;
}

static enkf_config_node_type *enkf_config_node_alloc__(enkf_var_type var_type,
                                                       ert_impl_type impl_type,
                                                       const char *key,
                                                       bool forward_init) {
    enkf_config_node_type *node =
        (enkf_config_node_type *)util_malloc(sizeof *node);
    node->forward_init = forward_init;
    node->var_type = var_type;
    node->impl_type = impl_type;
    node->key = util_alloc_string_copy(key);
    node->vector_storage = false;

    node->init_file_abs_path = NULL, node->init_file_fmt = NULL;
    node->enkf_infile_fmt = NULL;
    node->enkf_outfile_fmt = NULL;
    node->internalize = NULL;
    node->data = NULL;
    node->obs_keys = stringlist_alloc_new();

    node->get_data_size = NULL;
    node->freef = NULL;

    switch (impl_type) {
    case (FIELD):
        node->freef = field_config_free__;
        node->get_data_size = field_config_get_data_size__;
        break;
    case (GEN_KW):
        node->freef = gen_kw_config_free__;
        node->get_data_size = gen_kw_config_get_data_size__;
        break;
    case (SUMMARY):
        node->vector_storage = true;
        node->freef = summary_config_free__;
        node->get_data_size = summary_config_get_data_size__;
        break;
    case (GEN_DATA):
        node->freef = gen_data_config_free__;
        node->get_data_size = NULL;
        break;
    case (SURFACE):
        node->freef = surface_config_free__;
        node->get_data_size = surface_config_get_data_size__;
        break;
    case (EXT_PARAM):
        node->freef = ext_param_config_free__;
        node->get_data_size = ext_param_config_get_data_size__;
        break;
    default:
        util_abort("%s : invalid implementation type: %d - aborting \n",
                   __func__, impl_type);
    }
    return node;
}

bool enkf_config_node_vector_storage(const enkf_config_node_type *config_node) {
    return config_node->vector_storage;
}

static void enkf_config_node_update(enkf_config_node_type *config_node,
                                    const char *initfile_fmt,
                                    const char *enkf_outfile_fmt,
                                    const char *enkf_infile_fmt) {

    config_node->init_file_fmt =
        path_fmt_realloc_path_fmt(config_node->init_file_fmt, initfile_fmt);
    config_node->enkf_infile_fmt = path_fmt_realloc_path_fmt(
        config_node->enkf_infile_fmt, enkf_infile_fmt);
    config_node->enkf_outfile_fmt = path_fmt_realloc_path_fmt(
        config_node->enkf_outfile_fmt, enkf_outfile_fmt);
}

enkf_config_node_type *
enkf_config_node_alloc(enkf_var_type var_type, ert_impl_type impl_type,
                       bool forward_init, const char *key,
                       const char *init_file_fmt, const char *enkf_outfile_fmt,
                       const char *enkf_infile_fmt, void *data) {

    enkf_config_node_type *node =
        enkf_config_node_alloc__(var_type, impl_type, key, forward_init);
    enkf_config_node_update(node, init_file_fmt, enkf_outfile_fmt,
                            enkf_infile_fmt);
    node->data = data;
    return node;
}

void enkf_config_node_update_gen_kw(
    enkf_config_node_type *config_node,
    const char *
        enkf_outfile_fmt, /* The include file created by ERT for the forward model. */
    const char *template_file, const char *parameter_file,
    const char *init_file_fmt) {

    /* 1: Update the low level gen_kw_config stuff. */
    gen_kw_config_update((gen_kw_config_type *)config_node->data, template_file,
                         parameter_file);

    /* 2: Update the stuff which is owned by the upper-level enkf_config_node instance. */
    enkf_config_node_update(config_node, init_file_fmt, enkf_outfile_fmt, NULL);
}

/**
   This will create a new gen_kw_config instance which is NOT yet
   valid.
*/
enkf_config_node_type *enkf_config_node_new_gen_kw(const char *key,
                                                   const char *tag_fmt,
                                                   bool forward_init) {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc__(PARAMETER, GEN_KW, key, forward_init);
    config_node->data = gen_kw_config_alloc_empty(key, tag_fmt);
    return config_node;
}

enkf_config_node_type *enkf_config_node_new_surface(const char *key,
                                                    bool forward_init) {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc__(PARAMETER, SURFACE, key, forward_init);
    config_node->data = surface_config_alloc_empty();
    return config_node;
}

void enkf_config_node_update_surface(enkf_config_node_type *config_node,
                                     const char *base_surface,
                                     const char *init_file_fmt,
                                     const char *output_file) {

    /* 1: Update the data owned by the surface node. */
    surface_config_set_base_surface((surface_config_type *)config_node->data,
                                    base_surface);

    /* 2: Update the stuff which is owned by the upper-level enkf_config_node instance. */
    enkf_config_node_update(config_node, init_file_fmt, output_file, NULL);
}

enkf_config_node_type *
enkf_config_node_alloc_summary(const char *key, load_fail_type load_fail) {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc__(DYNAMIC_RESULT, SUMMARY, key, false);
    config_node->data = summary_config_alloc(key, load_fail);
    return config_node;
}

enkf_config_node_type *
enkf_config_node_alloc_GEN_DATA_everest(const char *key,
                                        const char *result_file_fmt,
                                        const int_vector_type *report_steps) {

    if (!gen_data_config_valid_result_format(result_file_fmt))
        return NULL;

    enkf_config_node_type *config_node =
        enkf_config_node_alloc_GEN_DATA_result(key, ASCII, result_file_fmt);
    gen_data_config_type *gen_data_config =
        (gen_data_config_type *)enkf_config_node_get_ref(config_node);

    for (int i = 0; i < int_vector_size(report_steps); i++) {
        int report_step = int_vector_iget(report_steps, i);
        gen_data_config_add_report_step(gen_data_config, report_step);
        enkf_config_node_set_internalize(config_node, report_step);
    }

    return config_node;
}

enkf_config_node_type *
enkf_config_node_alloc_GEN_DATA_result(const char *key,
                                       gen_data_file_format_type input_format,
                                       const char *enkf_infile_fmt) {

    enkf_config_node_type *config_node =
        enkf_config_node_alloc__(DYNAMIC_RESULT, GEN_DATA, key, false);
    config_node->data =
        gen_data_config_alloc_GEN_DATA_result(key, input_format);

    enkf_config_node_update(
        config_node, /* Generic update - needs the format settings from the special.*/
        NULL, NULL, enkf_infile_fmt);

    return config_node;
}

/**
   This will create a new gen_kw_config instance which is NOT yet
   valid. Mainly support code for the GUI.
*/
enkf_config_node_type *
enkf_config_node_alloc_field(const char *key, ecl_grid_type *ecl_grid,
                             field_trans_table_type *trans_table,
                             bool forward_init) {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc__(INVALID_VAR, FIELD, key, forward_init);
    config_node->data =
        field_config_alloc_empty(key, ecl_grid, trans_table, false);
    return config_node;
}

void enkf_config_node_update_parameter_field(enkf_config_node_type *config_node,
                                             const char *enkf_outfile_fmt,
                                             const char *init_file_fmt,
                                             int truncation, double value_min,
                                             double value_max,
                                             const char *init_transform,
                                             const char *output_transform) {

    field_file_format_type export_format = field_config_default_export_format(
        enkf_outfile_fmt); /* Purely based on extension, recognizes ROFF and GRDECL, the rest will be ecl_kw format. */
    field_config_update_parameter_field(
        (field_config_type *)config_node->data, truncation, value_min,
        value_max, export_format, init_transform, output_transform);
    config_node->var_type = PARAMETER;
    enkf_config_node_update(config_node, init_file_fmt, enkf_outfile_fmt, NULL);
}

void enkf_config_node_update_general_field(
    enkf_config_node_type *config_node, const char *enkf_outfile_fmt,
    const char *enkf_infile_fmt, const char *init_file_fmt, int truncation,
    double value_min, double value_max, const char *init_transform,
    const char *input_transform, const char *output_transform) {

    field_file_format_type export_format = field_config_default_export_format(
        enkf_outfile_fmt); /* Purely based on extension, recognizes ROFF and GRDECL, the rest will be ecl_kw format. */
    {
        enkf_var_type var_type = INVALID_VAR;
        if (enkf_infile_fmt == NULL)
            var_type = PARAMETER;
        else {
            if (enkf_outfile_fmt == NULL)
                var_type = DYNAMIC_RESULT; /* Probably not very realistic */
            else
                util_abort("%s: this used to be DYNAMIC_STATE ?? \n", __func__);
        }
        config_node->var_type = var_type;
    }
    field_config_update_general_field((field_config_type *)config_node->data,
                                      truncation, value_min, value_max,
                                      export_format, init_transform,
                                      input_transform, output_transform);

    enkf_config_node_update(config_node, init_file_fmt, enkf_outfile_fmt,
                            enkf_infile_fmt);
}

/**
   Invokes the get_data_size() function of the underlying node object.
*/
int enkf_config_node_get_data_size(const enkf_config_node_type *node,
                                   int report_step) {
    if (node->impl_type == GEN_DATA)
        return gen_data_config_get_data_size(
            (const gen_data_config_type *)node->data, report_step);
    else
        return node->get_data_size(node->data);
}

void enkf_config_node_free(enkf_config_node_type *node) {
    /* Freeing the underlying node object. */
    if (node->freef != NULL)
        node->freef(node->data);
    free(node->key);
    stringlist_free(node->obs_keys);

    free(node->init_file_abs_path);

    if (node->enkf_infile_fmt != NULL)
        path_fmt_free(node->enkf_infile_fmt);

    if (node->enkf_outfile_fmt != NULL)
        path_fmt_free(node->enkf_outfile_fmt);

    if (node->init_file_fmt != NULL)
        path_fmt_free(node->init_file_fmt);

    if (node->internalize != NULL)
        bool_vector_free(node->internalize);

    free(node);
}

const char *
enkf_config_node_get_enkf_outfile(const enkf_config_node_type *config_node) {
    return path_fmt_get_fmt(config_node->enkf_outfile_fmt);
}

const char *
enkf_config_node_get_enkf_infile(const enkf_config_node_type *config_node) {
    return path_fmt_get_fmt(config_node->enkf_infile_fmt);
}

const char *
enkf_config_node_get_init_file_fmt(const enkf_config_node_type *config_node) {
    return path_fmt_get_fmt(config_node->init_file_fmt);
}

/**
 * @brief Sets the given node to be internalized at the given report step
 *
 * Internalize means loaded from the forward simulation and stored in the
 * enkf_fs 'database'.
 *
 * @param node The config node to be internalized
 * @param report_step The report step for which the node should be internalized.
 */
void enkf_config_node_set_internalize(enkf_config_node_type *node,
                                      int report_step) {
    if (node->internalize == NULL)
        node->internalize = bool_vector_alloc(0, false);
    bool_vector_iset(node->internalize, report_step, true);
}

/** @return whether the given config node should be internalized at the given
 * report step.
 */
bool enkf_config_node_internalize(const enkf_config_node_type *node,
                                  int report_step) {
    if (node->internalize == NULL)
        return false;
    else
        return bool_vector_safe_iget(
            node->internalize,
            report_step); // Will return default value if report_step is beyond size.
}

/**
   This is the filename used when loading from a completed forward
   model.
*/
char *enkf_config_node_alloc_infile(const enkf_config_node_type *node,
                                    int report_step) {
    if (node->enkf_infile_fmt != NULL)
        return path_fmt_alloc_path(node->enkf_infile_fmt, false, report_step);
    else
        return NULL;
}

char *enkf_config_node_alloc_outfile(const enkf_config_node_type *node,
                                     int report_step) {
    if (node->enkf_outfile_fmt != NULL)
        return path_fmt_alloc_path(node->enkf_outfile_fmt, false, report_step);
    else
        return NULL;
}

/**
  The path argument is used when the function is during forward_model
  based initialisation.
*/
char *enkf_config_node_alloc_initfile(const enkf_config_node_type *node,
                                      const char *path, int iens) {
    if (node->init_file_fmt == NULL)
        return NULL;
    else {
        char *file = path_fmt_alloc_file(node->init_file_fmt, false, iens);
        if (util_is_abs_path(file))
            return file;
        else {
            char *full_path = util_alloc_filename(path, file, NULL);
            free(file);
            return full_path;
        }
    }
}

void *
enkf_config_node_get_ref(const enkf_config_node_type *node) { // CXX_CAST_ERROR
    return node->data;
}

bool enkf_config_node_include_type(const enkf_config_node_type *config_node,
                                   int mask) {

    enkf_var_type var_type = config_node->var_type;
    if (var_type & mask)
        return true;
    else
        return false;
}

bool enkf_config_node_use_forward_init(
    const enkf_config_node_type *config_node) {
    return config_node->forward_init;
}

ert_impl_type
enkf_config_node_get_impl_type(const enkf_config_node_type *config_node) {
    return config_node->impl_type;
}

enkf_var_type
enkf_config_node_get_var_type(const enkf_config_node_type *config_node) {
    return config_node->var_type;
}

const char *enkf_config_node_get_key(const enkf_config_node_type *config_node) {
    return config_node->key;
}

const stringlist_type *
enkf_config_node_get_obs_keys(const enkf_config_node_type *config_node) {
    return config_node->obs_keys;
}

int enkf_config_node_get_num_obs(const enkf_config_node_type *config_node) {
    return stringlist_get_size(config_node->obs_keys);
}

/**
   This checks the index_key - and sums up over all the time points of the observation.
*/
int enkf_config_node_load_obs(const enkf_config_node_type *config_node,
                              enkf_obs_type *enkf_obs, const char *key_index,
                              int obs_count, time_t *_sim_time, double *_y,
                              double *_std) {
    ert_impl_type impl_type = enkf_config_node_get_impl_type(config_node);
    int num_obs = 0;
    int iobs;

    for (iobs = 0; iobs < stringlist_get_size(config_node->obs_keys); iobs++) {
        obs_vector_type *obs_vector = enkf_obs_get_vector(
            enkf_obs, stringlist_iget(config_node->obs_keys, iobs));

        int report_step = -1;
        while (true) {
            report_step =
                obs_vector_get_next_active_step(obs_vector, report_step);
            if (report_step == -1)
                break;
            {
                bool valid;
                double value, std1;

                /*
                 * The user index used when calling the user_get function on the
                 * gen_obs data type is different depending on whether is called with a
                 * data context user_key (as here) or with a observation context
                 * user_key (as when plotting an observation plot). See more
                 * documentation of the function gen_obs_user_get_data_index().
                 */

                if (impl_type == GEN_DATA)
                    gen_obs_user_get_with_data_index(
                        (gen_obs_type *)obs_vector_iget_node(obs_vector,
                                                             report_step),
                        key_index, &value, &std1, &valid);
                else
                    obs_vector_user_get(obs_vector, key_index, report_step,
                                        &value, &std1, &valid);

                if (valid) {
                    if (obs_count > 0) {
                        _sim_time[num_obs] =
                            enkf_obs_iget_obs_time(enkf_obs, report_step);
                        _y[num_obs] = value;
                        _std[num_obs] = std1;
                    }
                    num_obs++;
                }
            }
        }
    }

    /* Sorting the observations in time order. */
    if (obs_count > 0) {
        double_vector_type *y =
            double_vector_alloc_shared_wrapper(0, 0, _y, obs_count);
        double_vector_type *std =
            double_vector_alloc_shared_wrapper(0, 0, _std, obs_count);
        time_t_vector_type *sim_time =
            time_t_vector_alloc_shared_wrapper(0, 0, _sim_time, obs_count);
        perm_vector_type *sort_perm = time_t_vector_alloc_sort_perm(sim_time);

        time_t_vector_permute(sim_time, sort_perm);
        double_vector_permute(y, sort_perm);
        double_vector_permute(std, sort_perm);

        free(sort_perm);
        double_vector_free(y);
        double_vector_free(std);
        time_t_vector_free(sim_time);
    }
    return num_obs;
}

void enkf_config_node_add_obs_key(enkf_config_node_type *config_node,
                                  const char *obs_key) {
    if (!stringlist_contains(config_node->obs_keys, obs_key))
        stringlist_append_copy(config_node->obs_keys, obs_key);
}

void enkf_config_node_clear_obs_keys(enkf_config_node_type *config_node) {
    stringlist_clear(config_node->obs_keys);
}

void enkf_config_node_add_GEN_DATA_config_schema(config_parser_type *config) {
    config_schema_item_type *item;
    item = config_add_schema_item(config, GEN_DATA_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);
}

enkf_config_node_type *enkf_config_node_alloc_GEN_DATA_from_config(
    const config_content_node_type *node) {
    enkf_config_node_type *config_node = NULL;
    const char *node_key = config_content_node_iget(node, 0);
    {
        hash_type *options = hash_alloc();

        config_content_node_init_opt_hash(node, options, 1);
        {
            gen_data_file_format_type input_format =
                gen_data_config_check_format(
                    (const char *)hash_safe_get(options, INPUT_FORMAT_KEY));
            const char *init_file_fmt =
                (const char *)hash_safe_get(options, INIT_FILES_KEY);
            const char *ecl_file =
                (const char *)hash_safe_get(options, ECL_FILE_KEY);
            const char *template_file =
                (const char *)hash_safe_get(options, TEMPLATE_KEY);
            const char *data_key =
                (const char *)hash_safe_get(options, KEY_KEY);
            const char *result_file =
                (const char *)hash_safe_get(options, RESULT_FILE_KEY);
            const char *forward_string =
                (const char *)hash_safe_get(options, FORWARD_INIT_KEY);
            const char *report_steps_string =
                (const char *)hash_safe_get(options, REPORT_STEPS_KEY);
            int_vector_type *report_steps = int_vector_alloc(0, 0);
            bool forward_init = false;
            bool valid_input = true;

            if (input_format == GEN_DATA_UNDEFINED)
                valid_input = false;

            if (!gen_data_config_valid_result_format(result_file)) {
                fprintf(
                    stderr,
                    "** ERROR: The RESULT_FILE:%s setting for %s is invalid - "
                    "must have an embedded %%d - and be a relative path.\n",
                    result_file, node_key);
                valid_input = false;
            }

            if (report_steps_string) {
                if (!string_util_update_active_list(report_steps_string,
                                                    report_steps)) {
                    valid_input = false;
                    fprintf(stderr,
                            "** ERROR: The REPORT_STEPS:%s attribute was not "
                            "valid.\n",
                            report_steps_string);
                }
            } else {
                fprintf(stderr, "** ERROR: As of July 2014 the GEN_DATA "
                                "keywords must have a REPORT_STEPS:xxxx \n");
                fprintf(stderr, "          attribute to indicate which report "
                                "step(s) you want to load data \n");
                fprintf(stderr, "          from. By requiring the user to "
                                "enter this information in advance\n");
                fprintf(stderr, "          it is easier for ERT for to check "
                                "that the results are valid, and\n");
                fprintf(stderr, "          handle errors with the GEN_DATA "
                                "results gracefully.\n");
                fprintf(stderr, "          \n");
                fprintf(stderr, "          You can list several report steps "
                                "separated with ',' and ranges with '-' \n");
                fprintf(stderr,
                        "          but observe that spaces is NOT ALLOWED. \n");
                fprintf(stderr, "          \n");
                fprintf(stderr, "           - load from report step 100:       "
                                "          REPORT_STEPS:100 \n");
                fprintf(stderr, "           - load from report steps 10, 20 "
                                "and 30-40    REPORT_STEPS:10,20,30-40 \n");
                fprintf(stderr, "          \n");
                fprintf(stderr,
                        "          The GEN_DATA keyword: %s will be ignored\n",
                        node_key);
                valid_input = false;
            }

            if (valid_input) {

                if (forward_string) {
                    if (!util_sscanf_bool(forward_string, &forward_init))
                        fprintf(stderr,
                                "** Warning: parsing %s as bool failed - using "
                                "FALSE \n",
                                forward_string);
                }

                if ((init_file_fmt == NULL) && (ecl_file == NULL) &&
                    (result_file != NULL))
                    config_node = enkf_config_node_alloc_GEN_DATA_result(
                        node_key, input_format, result_file);
                else if ((init_file_fmt != NULL) && (ecl_file != NULL) &&
                         (result_file != NULL))
                    util_abort(
                        "%s: This used to call the removed "
                        "enkf_config_node_alloc_GEN_DATA_state() function \n",
                        __func__);
                {
                    gen_data_config_type *gen_data_config =
                        (gen_data_config_type *)enkf_config_node_get_ref(
                            config_node);

                    if (template_file)
                        gen_data_config_set_template(gen_data_config,
                                                     template_file, data_key);

                    for (int i = 0; i < int_vector_size(report_steps); i++) {
                        int report_step = int_vector_iget(report_steps, i);
                        gen_data_config_add_report_step(gen_data_config,
                                                        report_step);
                        enkf_config_node_set_internalize(config_node,
                                                         report_step);
                    }
                }
            }

            int_vector_free(report_steps);
        }
        hash_free(options);
    }

    return config_node;
}

enkf_config_node_type *enkf_config_node_alloc_GEN_DATA_full(
    const char *node_key, const char *result_file,
    gen_data_file_format_type input_format, const int_vector_type *report_steps,
    const char *ecl_file, const char *init_file_fmt, const char *template_file,
    const char *data_key) {
    enkf_config_node_type *config_node = NULL;

    if ((init_file_fmt == NULL) && (ecl_file == NULL) &&
        (result_file != NULL)) {
        config_node = enkf_config_node_alloc_GEN_DATA_result(
            node_key, input_format, result_file);
    } else if ((init_file_fmt != NULL) && (ecl_file != NULL) &&
               (result_file != NULL)) {
        util_abort("%s: This used to call the removed "
                   "enkf_config_node_alloc_GEN_DATA_state() function \n",
                   __func__);
    }
    gen_data_config_type *gen_data_config =
        (gen_data_config_type *)enkf_config_node_get_ref(config_node);

    if (template_file)
        gen_data_config_set_template(gen_data_config, template_file, data_key);

    for (int i = 0; i < int_vector_size(report_steps); i++) {
        int report_step = int_vector_iget(report_steps, i);
        gen_data_config_add_report_step(gen_data_config, report_step);
        enkf_config_node_set_internalize(config_node, report_step);
    }

    return config_node;
}

enkf_config_node_type *enkf_config_node_alloc_GEN_KW_full(
    const char *node_key, bool forward_init, const char *gen_kw_format,
    const char *template_file, const char *enkf_outfile,
    const char *parameter_file, const char *init_file_fmt) {
    enkf_config_node_type *config_node = NULL;
    config_node =
        enkf_config_node_new_gen_kw(node_key, gen_kw_format, forward_init);

    enkf_config_node_update_gen_kw(config_node, enkf_outfile, template_file,
                                   parameter_file, init_file_fmt);

    return config_node;
}

enkf_config_node_type *enkf_config_node_alloc_SURFACE_full(
    const char *node_key, bool forward_init, const char *output_file,
    const char *base_surface, const char *init_file_fmt) {

    enkf_config_node_type *config_node =
        enkf_config_node_new_surface(node_key, forward_init);
    enkf_config_node_update_surface(config_node, base_surface, init_file_fmt,
                                    output_file);

    return config_node;
}
VOID_FREE(enkf_config_node)
