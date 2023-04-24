#include <filesystem>

#include <stdio.h>
#include <stdlib.h>

#include <ert/ecl/ecl_grid.h>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/ext_param_config.hpp>
#include <ert/enkf/gen_data_config.hpp>
#include <ert/enkf/gen_obs.hpp>
#include <ert/python.hpp>
#include <ert/res_util/path_fmt.hpp>
#include <ert/util/string_util.h>
#include <ert/util/stringlist.h>
#include <ert/util/vector.h>

namespace fs = std::filesystem;

static enkf_config_node_type *enkf_config_node_alloc__(ert_impl_type impl_type,
                                                       const char *key) {
    enkf_config_node_type *node =
        (enkf_config_node_type *)util_malloc(sizeof *node);
    node->impl_type = impl_type;
    node->key = util_alloc_string_copy(key);
    node->data = NULL;
    node->obs_keys = stringlist_alloc_new();

    node->get_data_size = NULL;
    node->freef = NULL;

    switch (impl_type) {
    case (SUMMARY):
        node->freef = summary_config_free__;
        node->get_data_size = summary_config_get_data_size__;
        break;
    case (GEN_DATA):
        node->freef = gen_data_config_free__;
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

enkf_config_node_type *enkf_config_node_alloc(ert_impl_type impl_type,
                                              const char *key, void *data) {

    enkf_config_node_type *node = enkf_config_node_alloc__(impl_type, key);
    node->data = data;
    return node;
}

enkf_config_node_type *
enkf_config_node_alloc_summary(const char *key, load_fail_type load_fail) {
    enkf_config_node_type *config_node = enkf_config_node_alloc__(SUMMARY, key);
    config_node->data = summary_config_alloc(key, load_fail);
    return config_node;
}

enkf_config_node_type *
enkf_config_node_alloc_GEN_DATA_everest(const char *key,
                                        const int_vector_type *report_steps) {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc_GEN_DATA_result(key, ASCII);
    gen_data_config_type *gen_data_config =
        (gen_data_config_type *)enkf_config_node_get_ref(config_node);

    for (int i = 0; i < int_vector_size(report_steps); i++) {
        int report_step = int_vector_iget(report_steps, i);
        gen_data_config_add_report_step(gen_data_config, report_step);
    }

    return config_node;
}

enkf_config_node_type *
enkf_config_node_alloc_GEN_DATA_result(const char *key,
                                       gen_data_file_format_type input_format) {

    enkf_config_node_type *config_node =
        enkf_config_node_alloc__(GEN_DATA, key);
    config_node->data =
        gen_data_config_alloc_GEN_DATA_result(key, input_format);
    return config_node;
}

void enkf_config_node_free(enkf_config_node_type *node) {
    /* Freeing the underlying node object. */
    if (node->freef != NULL)
        node->freef(node->data);
    free(node->key);
    stringlist_free(node->obs_keys);
    free(node);
}

void *
enkf_config_node_get_ref(const enkf_config_node_type *node) { // CXX_CAST_ERROR
    return node->data;
}

ert_impl_type
enkf_config_node_get_impl_type(const enkf_config_node_type *config_node) {
    return config_node->impl_type;
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
                                     gen_data_file_format_type input_format,
                                     const int_vector_type *report_steps) {
    enkf_config_node_type *config_node =
        enkf_config_node_alloc_GEN_DATA_result(node_key, input_format);
    gen_data_config_type *gen_data_config =
        (gen_data_config_type *)enkf_config_node_get_ref(config_node);

    for (int i = 0; i < int_vector_size(report_steps); i++) {
        int report_step = int_vector_iget(report_steps, i);
        gen_data_config_add_report_step(gen_data_config, report_step);
    }

    return config_node;
}

VOID_FREE(enkf_config_node)
