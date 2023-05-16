#include <filesystem>

#include <stdio.h>
#include <stdlib.h>

#include <ert/ecl/ecl_grid.h>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/ext_param_config.hpp>
#include <ert/enkf/gen_obs.hpp>
#include <ert/python.hpp>
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

    node->get_data_size = NULL;
    node->freef = NULL;

    switch (impl_type) {
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

void enkf_config_node_free(enkf_config_node_type *node) {
    /* Freeing the underlying node object. */
    if (node->freef != NULL)
        node->freef(node->data);
    free(node->key);
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

VOID_FREE(enkf_config_node)
