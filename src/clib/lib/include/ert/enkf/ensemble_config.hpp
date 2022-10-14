#ifndef ERT_ENSEMBLE_CONFIG_H
#define ERT_ENSEMBLE_CONFIG_H
#include <stdbool.h>

#include <string>
#include <vector>

#include <ert/util/hash.hpp>
#include <ert/util/stringlist.hpp>

#include <ert/ecl/ecl_grid.hpp>
#include <ert/ecl/ecl_sum.hpp>

#include <ert/config/config_content.hpp>
#include <ert/config/config_parser.hpp>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/summary_config.hpp>

typedef struct ensemble_config_struct ensemble_config_type;

extern "C" void ensemble_config_add_node(ensemble_config_type *ensemble_config,
                                         enkf_config_node_type *node);
extern "C" enkf_config_node_type *
ensemble_config_add_summary(ensemble_config_type *ensemble_config,
                            const char *key, load_fail_type load_fail);
enkf_config_node_type *
ensemble_config_add_summary_observation(ensemble_config_type *ensemble_config,
                                        const char *key,
                                        load_fail_type load_fail);
void ensemble_config_clear_obs_keys(ensemble_config_type *ensemble_config);
void ensemble_config_add_obs_key(ensemble_config_type *, const char *,
                                 const char *);
extern "C" void ensemble_config_free(ensemble_config_type *);
extern "C" bool ensemble_config_has_key(const ensemble_config_type *,
                                        const char *);
bool ensemble_config_have_forward_init(
    const ensemble_config_type *ensemble_config);
bool ensemble_config_require_summary(const ensemble_config_type *config);

extern "C" field_trans_table_type *
ensemble_config_get_trans_table(const ensemble_config_type *ensemble_config);
extern "C" enkf_config_node_type *
ensemble_config_get_node(const ensemble_config_type *, const char *);
enkf_config_node_type *ensemble_config_get_or_create_summary_node(
    ensemble_config_type *ensemble_config, const char *key);
extern "C" stringlist_type *
ensemble_config_alloc_keylist(const ensemble_config_type *);
std::vector<std::string>
ensemble_config_keylist_from_var_type(const ensemble_config_type *,
                                      int var_mask);
extern "C" stringlist_type *
ensemble_config_alloc_keylist_from_impl_type(const ensemble_config_type *,
                                             ert_impl_type);
extern "C" PY_USED ensemble_config_type *
ensemble_config_alloc_full(const char *gen_kw_format_string);
extern "C" void ensemble_config_init_SUMMARY_full(ensemble_config_type *,
                                                  const char *,
                                                  const ecl_sum_type *);

extern "C" int
ensemble_config_get_size(const ensemble_config_type *ensemble_config);
std::pair<fw_load_status, std::string>
ensemble_config_forward_init(const ensemble_config_type *ens_config,
                             const int iens, const std::string &run_path,
                             enkf_fs_type *sim_fs);

#endif
