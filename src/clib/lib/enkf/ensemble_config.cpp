#include "ert/python.hpp"
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <pthread.h> /* must have rw locking on the config_nodes ... */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ert/util/hash.h>
#include <ert/util/util.h>

#include <ert/res_util/string.hpp>

#include <ert/ecl/ecl_grid.h>

#include <ert/config/config_parser.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/gen_kw_config.hpp>
#include <ert/logging.hpp>

namespace fs = std::filesystem;

static auto logger = ert::get_logger("ensemble_config");

struct ensemble_config_struct {
    pthread_mutex_t mutex;
    char *
        gen_kw_format_string; /* format string used when creating gen_kw search/replace strings. */
    std::map<std::string, enkf_config_node_type *>
        config_nodes; /* a hash of enkf_config_node instances - which again contain pointers to e.g. field_config objects.  */
    field_trans_table_type *
        field_trans_table; /* a table of the transformations which are available to apply on fields. */
    bool have_forward_init;
    summary_key_matcher_type *summary_key_matcher;
};

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
void ensemble_config_set_gen_kw_format(ensemble_config_type *ensemble_config,
                                       const char *gen_kw_format_string) {
    if (!util_string_equal(gen_kw_format_string,
                           ensemble_config->gen_kw_format_string)) {
        stringlist_type *gen_kw_keys =
            ensemble_config_alloc_keylist_from_impl_type(ensemble_config,
                                                         GEN_KW);
        int i;
        ensemble_config->gen_kw_format_string = util_realloc_string_copy(
            ensemble_config->gen_kw_format_string, gen_kw_format_string);
        for (i = 0; i < stringlist_get_size(gen_kw_keys); i++) {
            enkf_config_node_type *config_node = ensemble_config_get_node(
                ensemble_config, stringlist_iget(gen_kw_keys, i));
            gen_kw_config_update_tag_format(
                (gen_kw_config_type *)enkf_config_node_get_ref(config_node),
                gen_kw_format_string); // CXX_CAST_ERROR
        }
        stringlist_free(gen_kw_keys);
    }
}

field_trans_table_type *
ensemble_config_get_trans_table(const ensemble_config_type *ensemble_config) {
    return ensemble_config->field_trans_table;
}

static ensemble_config_type *ensemble_config_alloc_empty(void) {
    ensemble_config_type *ensemble_config = new ensemble_config_type();

    ensemble_config->field_trans_table = field_trans_table_alloc();
    ensemble_config->gen_kw_format_string =
        util_alloc_string_copy(DEFAULT_GEN_KW_TAG_FORMAT);
    ensemble_config->have_forward_init = false;
    ensemble_config->summary_key_matcher = summary_key_matcher_alloc();
    pthread_mutex_init(&ensemble_config->mutex, NULL);

    return ensemble_config;
}

ensemble_config_type *
ensemble_config_alloc_full(const char *gen_kw_format_string) {
    ensemble_config_type *ensemble_config = ensemble_config_alloc_empty();
    ensemble_config_set_gen_kw_format(ensemble_config, gen_kw_format_string);
    pthread_mutex_init(&ensemble_config->mutex, NULL);
    return ensemble_config;
}

void ensemble_config_free(ensemble_config_type *ensemble_config) {
    field_trans_table_free(ensemble_config->field_trans_table);
    summary_key_matcher_free(ensemble_config->summary_key_matcher);
    free(ensemble_config->gen_kw_format_string);

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

bool ensemble_config_have_forward_init(
    const ensemble_config_type *ensemble_config) {
    return ensemble_config->have_forward_init;
}

void ensemble_config_add_node(ensemble_config_type *ensemble_config,
                              enkf_config_node_type *node) {
    if (node) {
        const char *key = enkf_config_node_get_key(node);
        if (ensemble_config_has_key(ensemble_config, key))
            util_abort("%s: a configuration object:%s has already been added - "
                       "aborting \n",
                       __func__, key);

        ensemble_config->config_nodes[key] = node;
        ensemble_config->have_forward_init |=
            enkf_config_node_use_forward_init(node);
    } else
        util_abort("%s: internal error - tried to add NULL node to ensemble "
                   "configuration \n",
                   __func__);
}

void ensemble_config_add_obs_key(ensemble_config_type *ensemble_config,
                                 const char *key, const char *obs_key) {
    enkf_config_node_type *node = ensemble_config->config_nodes.at(key);
    enkf_config_node_add_obs_key(node, obs_key);
}

void ensemble_config_clear_obs_keys(ensemble_config_type *ensemble_config) {
    for (auto &config_pair : ensemble_config->config_nodes) {
        enkf_config_node_type *config_node = config_pair.second;
        stringlist_clear(config_node->obs_keys);
    }
}

void ensemble_config_init_SUMMARY_full(ensemble_config_type *ensemble_config,
                                       const char *key,
                                       const ecl_sum_type *refcase) {
    summary_key_matcher_add_summary_key(ensemble_config->summary_key_matcher,
                                        key);
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
        } else {
            fprintf(stderr,
                    "** Warning: Cannot expand %s due to missing refcase file."
                    " Provide refcase file or add fully expanded SUMMARY key"
                    " to configuration\n",
                    key);
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

bool ensemble_config_has_impl_type(const ensemble_config_type *config,
                                   const ert_impl_type impl_type) {
    for (const auto &config_pair : config->config_nodes) {
        if (impl_type == enkf_config_node_get_impl_type(config_pair.second))
            return true;
    }

    return false;
}

bool ensemble_config_require_summary(const ensemble_config_type *ens_config) {
    return ensemble_config_has_impl_type(ens_config, SUMMARY);
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

enkf_config_node_type *
ensemble_config_add_summary_observation(ensemble_config_type *ensemble_config,
                                        const char *key,
                                        load_fail_type load_fail) {
    enkf_config_node_type *config_node =
        ensemble_config_add_summary(ensemble_config, key, load_fail);

    summary_key_matcher_add_summary_key(ensemble_config->summary_key_matcher,
                                        key);

    return config_node;
}

const summary_key_matcher_type *ensemble_config_get_summary_key_matcher(
    const ensemble_config_type *ensemble_config) {
    return ensemble_config->summary_key_matcher;
}

int ensemble_config_get_size(const ensemble_config_type *ensemble_config) {
    return ensemble_config->config_nodes.size();
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
        return self->have_forward_init;
    });
}
