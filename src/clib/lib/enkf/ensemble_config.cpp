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
#include <ert/enkf/gen_kw_config.hpp>
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

ensemble_config_type *
ensemble_config_alloc_full(const char *gen_kw_format_string) {
    ensemble_config_type *ensemble_config = new ensemble_config_type();

    if (strcmp(gen_kw_format_string, DEFAULT_GEN_KW_TAG_FORMAT) != 0) {
        stringlist_type *gen_kw_keys =
            ensemble_config_alloc_keylist_from_impl_type(ensemble_config,
                                                         GEN_KW);
        for (int i = 0; i < stringlist_get_size(gen_kw_keys); i++) {
            enkf_config_node_type *config_node = ensemble_config_get_node(
                ensemble_config, stringlist_iget(gen_kw_keys, i));
            gen_kw_config_update_tag_format(
                (gen_kw_config_type *)enkf_config_node_get_ref(config_node),
                gen_kw_format_string); // CXX_CAST_ERROR
        }
        stringlist_free(gen_kw_keys);
    }

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

enkf_config_node_type *ensemble_config_get_or_create_summary_node(
    ensemble_config_type *ensemble_config, const char *key) {
    if (ensemble_config->config_nodes.count(key) == 0)
        ensemble_config_add_summary(ensemble_config, key, LOAD_FAIL_SILENT);

    return ensemble_config_get_node(ensemble_config, key);
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

void ensemble_config_init_SUMMARY_full(ensemble_config_type *ensemble_config,
                                       const char *key,
                                       const ecl_sum_type *refcase) {
    ensemble_config->summary_keys.push_back(std::string(key));
    if (util_string_has_wildcard(key)) {
        if (refcase != NULL) {
            stringlist_type *keys = stringlist_alloc_new();
            ecl_sum_select_matching_general_var_list(
                refcase, key,
                keys); /* expanding the wildcard notation with help of the refcase. */
            int k;
            for (k = 0; k < stringlist_get_size(keys); k++)
                ensemble_config_add_summary(ensemble_config,
                                            stringlist_iget(keys, k),
                                            LOAD_FAIL_SILENT);
            stringlist_free(keys);
        }
    } else {
        ensemble_config_add_summary(ensemble_config, key, LOAD_FAIL_SILENT);
    }
}

stringlist_type *
ensemble_config_alloc_keylist(const ensemble_config_type *config) {
    stringlist_type *s = stringlist_alloc_new();
    for (const auto &config_pair : config->config_nodes)
        stringlist_append_copy(s, config_pair.first.c_str());
    return s;
}

/*
   observe that var_type here is an integer - naturally written as a
   sum of enkf_var_type values:

     ensemble_config_keylist_from_var_type( config , parameter + dynamic_state);

*/

std::vector<std::string>
ensemble_config_keylist_from_var_type(const ensemble_config_type *config,
                                      int var_mask) {
    std::vector<std::string> key_list;

    for (const auto &config_pair : config->config_nodes) {
        const std::string key = config_pair.first;
        enkf_var_type var_type =
            enkf_config_node_get_var_type(config_pair.second);

        if (var_type & var_mask)
            key_list.push_back(key);
    }

    return key_list;
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

/**
   this function ensures that object contains a node with 'key' and
   type == summary.

   if the @refcase pointer is different from NULL the key will be
   validated. keys which do not exist in the refcase will be ignored,
   a warning will be printed on stderr and the function will return
   NULL.
*/
enkf_config_node_type *
ensemble_config_add_summary(ensemble_config_type *ensemble_config,
                            const char *key, load_fail_type load_fail) {
    enkf_config_node_type *config_node = NULL;

    const auto node_it = ensemble_config->config_nodes.find(key);
    if (node_it != ensemble_config->config_nodes.end()) {
        config_node = node_it->second;
        if (enkf_config_node_get_impl_type(config_node) != SUMMARY) {
            util_abort("%s: ensemble key:%s already exists - but it is not of "
                       "summary type\n",
                       __func__, key);
        }

        summary_config_type *summary_config =
            (summary_config_type *)enkf_config_node_get_ref(config_node);
        summary_config_update_load_fail_mode(summary_config, load_fail);

    } else {
        config_node = enkf_config_node_alloc_summary(key, load_fail);
        ensemble_config_add_node(ensemble_config, config_node);
    }

    return config_node;
}

std::pair<fw_load_status, std::string>
ensemble_config_forward_init(const ensemble_config_type *ens_config,
                             const int iens, const std::string &run_path,
                             enkf_fs_type *sim_fs) {
    auto result = LOAD_SUCCESSFUL;
    std::string error_msg;
    {
        for (auto &config_pair : ens_config->config_nodes) {
            enkf_config_node_type *config_node = config_pair.second;
            if (enkf_config_node_use_forward_init(config_node)) {
                enkf_node_type *node = enkf_node_alloc(config_node);
                node_id_type node_id = {.report_step = 0, .iens = iens};

                if (!enkf_node_has_data(node, sim_fs, node_id)) {
                    if (enkf_node_forward_init(node, run_path, iens))
                        enkf_node_store(node, sim_fs, node_id);
                    else {
                        char *init_file = enkf_config_node_alloc_initfile(
                            enkf_node_get_config(node), run_path.c_str(), iens);

                        if (init_file && !fs::exists(init_file))
                            error_msg = fmt::format(
                                "File not found: {} - failed to initialize "
                                "node: {}\n",
                                init_file, enkf_node_get_key(node));
                        else
                            error_msg =
                                fmt::format("Failed to initialize node: {}\n",
                                            enkf_node_get_key(node));

                        free(init_file);
                        result = LOAD_FAILURE;
                    }
                }
                enkf_node_free(node);
            }
        }
    }
    return {result, error_msg};
}

ERT_CLIB_SUBMODULE("ensemble_config", m) {
    m.def("have_forward_init", [](Cwrap<ensemble_config_type> self) {
        for (auto const &[node_key, enkf_config_node] : self->config_nodes) {
            if (enkf_config_node->forward_init)
                return true;
        }

        return false;
    });
    m.def("get_summary_keys",
          [](Cwrap<ensemble_config_type> self) { return self->summary_keys; });
}
