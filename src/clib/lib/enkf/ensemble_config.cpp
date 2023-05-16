#include "ert/python.hpp"
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ert/util/hash.h>
#include <ert/util/util.h>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/logging.hpp>

namespace fs = std::filesystem;
/**
   setting the format string used to 'mangle' the string in the gen_kw
   template files. consider the following example:

      parameter file
      --------------
      multpv   logunif  0.0001 0.10


      template file
      -------------
      box
         1  10  1 10  1 5 /

      multpv  500*__multpv__

   here the parameter file defines a parameter named 'multpv', and the
   template file uses the marker string '__multpv__' which should be
   replaced with a numerical value. for the current example the
   gen_kw_format_string should have the value '__%s__'.

   there are no rules for the format string, but it _must_ contain a
   '%s' placeholder which will be replaced with the parameter name
   (this is not checked for). the function call creating a search
   string from a parameter name is:

      tagged_string = util_alloc_sprintf( gen_kw_format_string , parameter_name );

*/

ensemble_config_type *ensemble_config_alloc_full() {
    ensemble_config_type *ensemble_config = new ensemble_config_type();
    return ensemble_config;
}

void ensemble_config_free(ensemble_config_type *ensemble_config) {
    for (auto &config_pair : ensemble_config->config_nodes)
        enkf_config_node_free(config_pair.second);

    delete ensemble_config;
}

bool ensemble_config_has_key(const ensemble_config_type *ensemble_config,
                             const char *key) {
    return ensemble_config->config_nodes.count(key) > 0;
}

enkf_config_node_type *
ensemble_config_get_node(const ensemble_config_type *ensemble_config,
                         const char *key) {
    const auto node_it = ensemble_config->config_nodes.find(key);
    if (node_it != ensemble_config->config_nodes.end()) {
        enkf_config_node_type *node = node_it->second;
        return node;
    } else {
        util_abort("%s: ens node:\"%s\" does not exist \n", __func__, key);
        return NULL; /* compiler shut up */
    }
}

void ensemble_config_add_node(ensemble_config_type *ensemble_config,
                              enkf_config_node_type *node) {
    if (node)
        ensemble_config->config_nodes[node->key] = node;
    else
        fprintf(
            stderr,
            "** Warning: Invalid node passed to ensemble_config_add_node\n");
}

std::vector<std::string>
ensemble_config_get_summary_key_list(const char *key,
                                     const ecl_sum_type *refcase) {
    std::vector<std::string> keylist;
    if (util_string_has_wildcard(key)) {
        if (refcase != nullptr) {
            stringlist_type *keys = stringlist_alloc_new();
            ecl_sum_select_matching_general_var_list(
                refcase, key,
                keys); /* expanding the wildcard notation with help of the refcase. */
            for (int k = 0; k < stringlist_get_size(keys); k++) {
                keylist.push_back((std::string)stringlist_iget(keys, k));
            }
            stringlist_free(keys);
        }
    } else {
        keylist.push_back(std::string(key));
    }

    return keylist;
}

stringlist_type *
ensemble_config_alloc_keylist(const ensemble_config_type *config) {
    stringlist_type *s = stringlist_alloc_new();
    for (const auto &config_pair : config->config_nodes)
        stringlist_append_copy(s, config_pair.first.c_str());
    return s;
}

stringlist_type *
ensemble_config_alloc_keylist_from_impl_type(const ensemble_config_type *config,
                                             ert_impl_type impl_type) {
    stringlist_type *key_list = stringlist_alloc_new();

    for (const auto &config_pair : config->config_nodes) {
        const char *key = config_pair.first.c_str();
        if (impl_type == enkf_config_node_get_impl_type(config_pair.second))
            stringlist_append_copy(key_list, key);
    }

    return key_list;
}

ERT_CLIB_SUBMODULE("ensemble_config", m) {
    m.def("get_summary_key_list",
          [](const char *key, Cwrap<ecl_sum_type> refcase) {
              return ensemble_config_get_summary_key_list(key, refcase);
          });
}
