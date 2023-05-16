#ifndef ERT_ENSEMBLE_CONFIG_H
#define ERT_ENSEMBLE_CONFIG_H
#include <stdbool.h>

#include <string>
#include <vector>

#include <ert/util/hash.hpp>
#include <ert/util/stringlist.hpp>
#include <map>

#include <ert/ecl/ecl_grid.hpp>
#include <ert/ecl/ecl_sum.hpp>

#include <ert/config/config_content.hpp>
#include <ert/config/config_parser.hpp>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_types.hpp>

typedef struct ensemble_config_struct ensemble_config_type;

struct ensemble_config_struct {
    std::map<std::string, enkf_config_node_type *>
        config_nodes; /* a hash of enkf_config_node instances - which again contain pointers to e.g. field_config objects.  */
};

extern "C" void ensemble_config_add_node(ensemble_config_type *ensemble_config,
                                         enkf_config_node_type *node);
extern "C" void ensemble_config_free(ensemble_config_type *);
extern "C" bool ensemble_config_has_key(const ensemble_config_type *,
                                        const char *);
extern "C" enkf_config_node_type *
ensemble_config_get_node(const ensemble_config_type *, const char *);
extern "C" stringlist_type *
ensemble_config_alloc_keylist(const ensemble_config_type *);

extern "C" stringlist_type *
ensemble_config_alloc_keylist_from_impl_type(const ensemble_config_type *,
                                             ert_impl_type);
extern "C" PY_USED ensemble_config_type *ensemble_config_alloc_full();
#endif
