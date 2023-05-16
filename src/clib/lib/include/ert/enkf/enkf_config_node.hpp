#ifndef ERT_ENKF_CONFIG_NODE_H
#define ERT_ENKF_CONFIG_NODE_H

#include <ert/util/hash.h>
#include <ert/util/stringlist.h>

#include <ert/config/config_parser.hpp>

#include <ert/ecl/ecl_grid.hpp>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/res_util/path_fmt.hpp>

typedef void(config_free_ftype)(void *);
typedef int(get_data_size_ftype)(const void *);
typedef void(config_fprintf_ftype)(const void *, FILE *);

typedef struct enkf_config_node_struct enkf_config_node_type;

struct enkf_config_node_struct {
    ert_impl_type impl_type;

    char *key;
    /** This points to the config object of the actual implementation. */
    void *data;

    /** Function pointer to ask the underlying config object of the size - i.e.
     * number of elements. */
    get_data_size_ftype *get_data_size;
    config_free_ftype *freef;
};

extern "C" enkf_config_node_type *enkf_config_node_alloc(ert_impl_type,
                                                         const char *, void *);

extern "C" void enkf_config_node_free(enkf_config_node_type *);
extern "C" ert_impl_type
enkf_config_node_get_impl_type(const enkf_config_node_type *);
extern "C" void *enkf_config_node_get_ref(const enkf_config_node_type *);
extern "C" const char *enkf_config_node_get_key(const enkf_config_node_type *);
/*
    The enkf_node_free() function declaration is in the enkf_config_node.h header,
    because the enkf_config_node needs to know how to free the min_std node.
    That's not really the reason, because min_std doesn't exist anymore, but
    the function declaration is needed anyways.
  */

VOID_FREE_HEADER(enkf_config_node);
#endif
