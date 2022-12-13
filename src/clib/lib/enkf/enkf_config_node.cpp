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
#include <ert/enkf/gen_data_config.hpp>
#include <ert/enkf/gen_kw_config.hpp>
#include <ert/enkf/gen_obs.hpp>
#include <ert/enkf/surface_config.hpp>
#include <ert/python.hpp>
#include <ert/res_util/path_fmt.hpp>
#include <ert/util/string_util.h>
#include <ert/util/stringlist.h>
#include <ert/util/vector.h>

namespace fs = std::filesystem;

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

    node->init_file_fmt = NULL;
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
        node->freef = summary_config_free__;
        node->get_data_size = summary_config_get_data_size__;
        break;
    case (GEN_DATA):
        node->freef = gen_data_config_free__;
        node->get_data_size = NULL;
        break;
    case (SURFACE):
        node->freef =
            reinterpret_cast<config_free_ftype *>(surface_config_free);
        node->get_data_size = NULL;
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
enkf_config_node_type *enkf_config_node_alloc_field(const char *key,
                                                    ecl_grid_type *ecl_grid,
                                                    bool forward_init) {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc__(INVALID_VAR, FIELD, key, forward_init);
    config_node->data = field_config_alloc_empty(key, ecl_grid, false);
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

enkf_config_node_type *
enkf_config_node_alloc_GEN_DATA_full(const char *node_key,
                                     const char *result_file,
                                     gen_data_file_format_type input_format,
                                     const int_vector_type *report_steps) {
    enkf_config_node_type *config_node = NULL;

    if (result_file != NULL) {
        config_node = enkf_config_node_alloc_GEN_DATA_result(
            node_key, input_format, result_file);
    }
    gen_data_config_type *gen_data_config =
        (gen_data_config_type *)enkf_config_node_get_ref(config_node);

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

    enkf_config_node_type *config_node =
        enkf_config_node_alloc__(PARAMETER, GEN_KW, node_key, forward_init);
    config_node->data = gen_kw_config_alloc_empty(node_key, gen_kw_format);

    /* 1: Update the low level gen_kw_config stuff. */
    gen_kw_config_update((gen_kw_config_type *)config_node->data, template_file,
                         parameter_file);

    /* 2: Update the stuff which is owned by the upper-level enkf_config_node instance. */
    enkf_config_node_update(config_node, init_file_fmt, enkf_outfile, NULL);

    return config_node;
}

enkf_config_node_type *enkf_config_node_alloc_SURFACE_full(
    const char *node_key, bool forward_init, const char *output_file,
    const char *base_surface, const char *init_file_fmt) {

    enkf_config_node_type *config_node =
        enkf_config_node_alloc__(PARAMETER, SURFACE, node_key, forward_init);
    config_node->data = surface_config_alloc_empty();

    /* 1: Update the data owned by the surface node. */
    surface_config_set_base_surface((surface_config_type *)config_node->data,
                                    base_surface);

    /* 2: Update the stuff which is owned by the upper-level enkf_config_node instance. */
    enkf_config_node_update(config_node, init_file_fmt, output_file, NULL);

    return config_node;
}
VOID_FREE(enkf_config_node)

ERT_CLIB_SUBMODULE("enkf_config_node", m) {
    m.def("alloc_outfile",
          [](Cwrap<enkf_config_node_type> self, int iens) -> py::object {
              char *path = nullptr;
              if (self->enkf_outfile_fmt != NULL)
                  path =
                      path_fmt_alloc_path(self->enkf_outfile_fmt, false, iens);

              if (path == nullptr)
                  return py::none{};
              else
                  return py::str{path};
          });
}
