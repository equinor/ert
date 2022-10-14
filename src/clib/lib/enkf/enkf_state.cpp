#include <stdexcept>
#include <string>
#include <vector>

#include <ert/python.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.h>

#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_sum.h>

#include "ert/enkf/ensemble_config.hpp"

#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/gen_data.hpp>
#include <ert/enkf/summary.hpp>
#include <ert/logging.hpp>
#include <ert/res_util/memory.hpp>

static auto logger = ert::get_logger("enkf");

void enkf_state_initialize(enkf_fs_type *fs, enkf_node_type *param_node,
                           int iens) {
    node_id_type node_id = {.report_step = 0, .iens = iens};
    if (enkf_node_initialize(param_node, iens))
        enkf_node_store(param_node, fs, node_id);
}

/**
 * Check if there are summary keys in the ensemble config that is not found in
 * Eclipse. If this is the case, AND we have observations for this key, we have
 * a problem. Otherwise, just print a message to the log.
 */
static std::pair<fw_load_status, std::string>
enkf_state_check_for_missing_eclipse_summary_data(
    const ensemble_config_type *ens_config,
    const std::vector<std::string> summary_keys, const ecl_smspec_type *smspec,
    const int iens) {

    std::pair<fw_load_status, std::string> result = {LOAD_SUCCESSFUL, ""};
    std::vector<std::string> missing_keys;
    for (auto summary_key : summary_keys) {

        const char *key = summary_key.c_str();

        if (ecl_smspec_has_general_var(smspec, key) ||
            !summary_key_matcher_summary_key_is_required(summary_keys,
                                                         summary_key))
            continue;

        if (!ensemble_config_has_key(ens_config, key))
            continue;

        const enkf_config_node_type *config_node =
            ensemble_config_get_node(ens_config, key);

        if (stringlist_get_size(config_node->obs_keys) == 0) {
            logger->info(
                "[{:03d}:----] Unable to find Eclipse data for summary key: "
                "{}, but have no observations either, so will continue.",
                iens, key);
        } else {
            missing_keys.push_back(summary_key);
        }
    }
    if (!missing_keys.empty())
        return {
            LOAD_FAILURE,
            fmt::format("Missing Eclipse data for required summary keys: {}",
                        fmt::join(missing_keys, ", "))};
    return result;
}

static std::pair<fw_load_status, std::string>
enkf_state_internalize_dynamic_eclipse_results(
    ensemble_config_type *ens_config, const ecl_sum_type *summary,
    const std::vector<std::string> summary_keys, enkf_fs_type *sim_fs,
    const int iens) {

    auto &time_map = enkf_fs_get_time_map(sim_fs);
    auto status = time_map.summary_update(summary);
    if (!status.empty()) {
        // Something has gone wrong in checking time map, fail
        return {TIME_MAP_FAILURE, status};
    }
    auto time_index = time_map.indices(summary);

    // The actual loading internalizing - from ecl_sum -> enkf_node.
    // step2 is just taken from the number of steps found in the
    // summary file.
    const int step2 = ecl_sum_get_last_report_step(summary);

    time_index[0] = -1; // don't load 0th index
    time_index.resize(step2 + 1, -1);

    const ecl_smspec_type *smspec = ecl_sum_get_smspec(summary);

    for (int i = 0; i < ecl_smspec_num_nodes(smspec); i++) {
        const ecl::smspec_node &smspec_node =
            ecl_smspec_iget_node_w_node_index(smspec, i);
        const char *key = smspec_node.get_gen_key1();

        if (summary_key_matcher_match_summary_key(summary_keys, key)) {

            enkf_config_node_type *config_node =
                ensemble_config_get_or_create_summary_node(ens_config, key);
            enkf_node_type *node = enkf_node_alloc(config_node);

            // Ensure that what is currently on file is loaded
            // before we update.
            enkf_node_try_load_vector(node, sim_fs, iens);

            summary_forward_load_vector(
                static_cast<summary_type *>(enkf_node_value_ptr(node)), summary,
                time_index);
            enkf_node_store_vector(node, sim_fs, iens);
            enkf_node_free(node);
        }
    }

    // Check if some of the specified keys are missing from the Eclipse
    // data, and if there are observations for them. That is a problem.
    return enkf_state_check_for_missing_eclipse_summary_data(
        ens_config, summary_keys, smspec, iens);
}

static fw_load_status enkf_state_load_gen_data_node(
    const std::string run_path, enkf_fs_type *sim_fs, int iens,
    const enkf_config_node_type *config_node, size_t start, size_t stop) {
    fw_load_status status = LOAD_SUCCESSFUL;
    for (int report_step = start; report_step <= stop; report_step++) {

        bool should_internalize = false;

        if (config_node->internalize != NULL)
            should_internalize =
                bool_vector_safe_iget(config_node->internalize, report_step);

        if (!should_internalize)
            continue;

        enkf_node_type *node = enkf_node_alloc(config_node);

        if (enkf_node_forward_load(node, report_step, run_path, sim_fs)) {
            node_id_type node_id = {.report_step = report_step, .iens = iens};

            enkf_node_store(node, sim_fs, node_id);
            logger->info("Loaded GEN_DATA: {} instance for step: {} from file: "
                         "{} size: {}",
                         enkf_node_get_key(node), report_step,
                         enkf_config_node_alloc_infile(
                             enkf_node_get_config(node), report_step),
                         gen_data_get_size(
                             (const gen_data_type *)enkf_node_value_ptr(node)));
        } else {
            logger->error(
                "[{:03d}:{:04d}] Failed load data for GEN_DATA node:{}.", iens,
                report_step, enkf_node_get_key(node));
            status = LOAD_FAILURE;
        }
        enkf_node_free(node);
    }
    return status;
}

static fw_load_status enkf_state_internalize_GEN_DATA(
    const ensemble_config_type *ens_config, const int iens,
    enkf_fs_type *sim_fs, const std::string &run_path, size_t num_reports) {

    stringlist_type *keylist_GEN_DATA =
        ensemble_config_alloc_keylist_from_impl_type(ens_config, GEN_DATA);

    int numkeys = stringlist_get_size(keylist_GEN_DATA);

    if (numkeys > 0)
        if (num_reports != 0)
            logger->warning(
                "Trying to load GEN_DATA without properly "
                "set num_reports (was {}) - will only look for step 0 data: {}",
                num_reports, stringlist_iget(keylist_GEN_DATA, 0));

    fw_load_status result = LOAD_SUCCESSFUL;
    for (int ikey = 0; ikey < numkeys; ikey++) {
        const enkf_config_node_type *config_node = ensemble_config_get_node(
            ens_config, stringlist_iget(keylist_GEN_DATA, ikey));

        // This for loop should probably be changed to use the report
        // steps configured in the gen_data_config object, instead of
        // spinning through them all.
        size_t start = 0;
        size_t stop = num_reports;
        auto status = enkf_state_load_gen_data_node(run_path, sim_fs, iens,
                                                    config_node, start, stop);
        if (status == LOAD_FAILURE)
            result = LOAD_FAILURE;
    }
    stringlist_free(keylist_GEN_DATA);
    return result;
}

ERT_CLIB_SUBMODULE("enkf_state", m) {
    m.def("state_initialize",
          [](Cwrap<enkf_node_type> param_node, Cwrap<enkf_fs_type> fs,
             int iens) { return enkf_state_initialize(fs, param_node, iens); });
    m.def("internalize_dynamic_eclipse_results",
          [](Cwrap<ensemble_config_type> ens_config,
             Cwrap<ecl_sum_type> summary, std::vector<std::string> summary_keys,
             Cwrap<enkf_fs_type> sim_fs, int iens) {
              return enkf_state_internalize_dynamic_eclipse_results(
                  ens_config, summary, summary_keys, sim_fs, iens);
          });
    m.def("internalize_GEN_DATA",
          [](Cwrap<ensemble_config_type> ens_config, Cwrap<enkf_fs_type> fs,
             int iens, const std::string &run_path, size_t num_reports) {
              return enkf_state_internalize_GEN_DATA(ens_config, iens, fs,
                                                     run_path, num_reports);
          });
}
