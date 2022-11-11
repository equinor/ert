#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <vector>

#include <ert/python.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.h>

#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_sum.h>

#include <ert/job_queue/environment_varlist.hpp>

#include "ert/enkf/ensemble_config.hpp"
#include "ert/enkf/model_config.hpp"
#include "ert/enkf/run_arg_type.hpp"

#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/gen_data.hpp>
#include <ert/logging.hpp>
#include <ert/res_util/memory.hpp>

static auto logger = ert::get_logger("enkf");

void enkf_state_initialize(enkf_fs_type *fs, enkf_node_type *param_node,
                           int iens) {
    node_id_type node_id = {.report_step = 0, .iens = iens};
    if (enkf_node_initialize(param_node, iens))
        enkf_node_store(param_node, fs, node_id);
}

ecl_sum_type *load_ecl_sum(const char *run_path, const char *eclbase) {
    ecl_sum_type *summary = NULL;

    char *header_file = ecl_util_alloc_exfilename(
        run_path, eclbase, ECL_SUMMARY_HEADER_FILE, DEFAULT_FORMATTED, -1);
    char *unified_file = ecl_util_alloc_exfilename(
        run_path, eclbase, ECL_UNIFIED_SUMMARY_FILE, DEFAULT_FORMATTED, -1);
    stringlist_type *data_files = stringlist_alloc_new();
    if ((unified_file != NULL) && (header_file != NULL)) {
        stringlist_append_copy(data_files, unified_file);

        bool include_restart = false;

        /*
             * Setting this flag causes summary-data to be loaded by
             * ecl::unsmry_loader which is "horribly slow" according
             * to comments in the code. The motivation for introducing
             * this mode was at some point to use less memory, but
             * computers nowadays should not have a problem with that.
             *
             * For comments, reasoning and discussions, please refer to
             * https://github.com/equinor/ert/issues/2873
             *   and
             * https://github.com/equinor/ert/issues/2972
             */
        bool lazy_load = false;
        if (std::getenv("ERT_LAZY_LOAD_SUMMARYDATA"))
            lazy_load = true;

        {
            ert::utils::scoped_memory_logger memlogger(
                logger, fmt::format("lazy={}", lazy_load));

            int file_options = 0;
            summary = ecl_sum_fread_alloc(
                header_file, data_files, SUMMARY_KEY_JOIN_STRING,
                include_restart, lazy_load, file_options);
        }
    } else {
        stringlist_free(data_files);
        throw std::invalid_argument(
            "Could not find SUMMARY file or using non unified SUMMARY file");
    }
    stringlist_free(data_files);
    free(header_file);
    free(unified_file);
    return summary;
}

/**
 * Check if there are summary keys in the ensemble config that is not found in
 * Eclipse. If this is the case, AND we have observations for this key, we have
 * a problem. Otherwise, just print a message to the log.
 */
static std::pair<fw_load_status, std::string>
enkf_state_check_for_missing_eclipse_summary_data(
    const ensemble_config_type *ens_config,
    const summary_key_matcher_type *matcher, const ecl_smspec_type *smspec,
    const int iens) {
    stringlist_type *keys = summary_key_matcher_get_keys(matcher);
    std::pair<fw_load_status, std::string> result = {LOAD_SUCCESSFUL, ""};
    std::vector<std::string> missing_keys;
    for (int i = 0; i < stringlist_get_size(keys); i++) {

        const char *key = stringlist_iget(keys, i);

        if (ecl_smspec_has_general_var(smspec, key) ||
            !summary_key_matcher_summary_key_is_required(matcher, key))
            continue;

        if (!ensemble_config_has_key(ens_config, key))
            continue;

        const enkf_config_node_type *config_node =
            ensemble_config_get_node(ens_config, key);
        if (enkf_config_node_get_num_obs(config_node) == 0) {
            logger->info(
                "[{:03d}:----] Unable to find Eclipse data for summary key: "
                "{}, but have no observations either, so will continue.",
                iens, key);
        } else {
            missing_keys.push_back(key);
        }
    }
    stringlist_free(keys);
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
    const summary_key_matcher_type *matcher, enkf_fs_type *sim_fs,
    const int iens) {
    int load_start = 0;

    if (load_start == 0) {
        // Do not attempt to load the "S0000" summary results.
        load_start++;
    }

    time_map_type *time_map = enkf_fs_get_time_map(sim_fs);
    auto status = time_map_summary_update(time_map, summary);
    if (!status.empty()) {
        // Something has gone wrong in checking time map, fail
        return {TIME_MAP_FAILURE, status};
    }
    int_vector_type *time_index = time_map_alloc_index_map(time_map, summary);

    // The actual loading internalizing - from ecl_sum -> enkf_node.
    // step2 is just taken from the number of steps found in the
    // summary file.
    const int step2 = ecl_sum_get_last_report_step(summary);

    int_vector_iset_block(time_index, 0, load_start, -1);
    int_vector_resize(time_index, step2 + 1, -1);

    const ecl_smspec_type *smspec = ecl_sum_get_smspec(summary);

    for (int i = 0; i < ecl_smspec_num_nodes(smspec); i++) {
        const ecl::smspec_node &smspec_node =
            ecl_smspec_iget_node_w_node_index(smspec, i);
        const char *key = smspec_node.get_gen_key1();

        if (summary_key_matcher_match_summary_key(matcher, key)) {
            summary_key_set_type *key_set = enkf_fs_get_summary_key_set(sim_fs);
            summary_key_set_add_summary_key(key_set, key);

            enkf_config_node_type *config_node =
                ensemble_config_get_or_create_summary_node(ens_config, key);
            enkf_node_type *node = enkf_node_alloc(config_node);

            // Ensure that what is currently on file is loaded
            // before we update.
            enkf_node_try_load_vector(node, sim_fs, iens);

            enkf_node_forward_load_vector(node, summary, time_index);
            enkf_node_store_vector(node, sim_fs, iens);
            enkf_node_free(node);
        }
    }
    int_vector_free(time_index);

    // Check if some of the specified keys are missing from the Eclipse
    // data, and if there are observations for them. That is a problem.
    return enkf_state_check_for_missing_eclipse_summary_data(
        ens_config, matcher, smspec, iens);
    return {LOAD_SUCCESSFUL, ""};
}

static fw_load_status enkf_state_load_gen_data_node(
    const run_arg_type *run_arg, enkf_fs_type *sim_fs, int iens,
    const enkf_config_node_type *config_node, int start, int stop) {
    fw_load_status status = LOAD_SUCCESSFUL;
    for (int report_step = start; report_step <= stop; report_step++) {
        if (!enkf_config_node_internalize(config_node, report_step))
            continue;

        enkf_node_type *node = enkf_node_alloc(config_node);

        if (enkf_node_forward_load(node, report_step,
                                   run_arg_get_runpath(run_arg),
                                   run_arg_get_sim_fs(run_arg))) {
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
    const ensemble_config_type *ens_config, const run_arg_type *run_arg,
    const model_config_type *model_config, int last_report) {

    stringlist_type *keylist_GEN_DATA =
        ensemble_config_alloc_keylist_from_impl_type(ens_config, GEN_DATA);

    int numkeys = stringlist_get_size(keylist_GEN_DATA);

    if (numkeys > 0)
        if (last_report <= 0)
            logger->warning(
                "Trying to load GEN_DATA without properly "
                "set last_report (was {}) - will only look for step 0 data: {}",
                last_report, stringlist_iget(keylist_GEN_DATA, 0));

    enkf_fs_type *sim_fs = run_arg_get_sim_fs(run_arg);
    const int iens = run_arg_get_iens(run_arg);
    fw_load_status result = LOAD_SUCCESSFUL;
    for (int ikey = 0; ikey < numkeys; ikey++) {
        const enkf_config_node_type *config_node = ensemble_config_get_node(
            ens_config, stringlist_iget(keylist_GEN_DATA, ikey));

        // This for loop should probably be changed to use the report
        // steps configured in the gen_data_config object, instead of
        // spinning through them all.
        int start = 0;
        int stop = util_int_max(0, last_report); // inclusive
        auto status = enkf_state_load_gen_data_node(run_arg, sim_fs, iens,
                                                    config_node, start, stop);
        if (status == LOAD_FAILURE)
            result = LOAD_FAILURE;
    }
    stringlist_free(keylist_GEN_DATA);
    return result;
}

/**
   This function loads the results from a forward simulations from report_step1
   to report_step2. The details of what to load are in model_config and the
   spesific nodes for special cases.

   Will mainly be called at the end of the forward model, but can also
   be called manually from external scope.
*/
static std::pair<fw_load_status, std::string>
enkf_state_internalize_results(ensemble_config_type *ens_config,
                               model_config_type *model_config,
                               const run_arg_type *run_arg) {
    const summary_key_matcher_type *matcher =
        ensemble_config_get_summary_key_matcher(ens_config);

    if (summary_key_matcher_get_size(matcher) > 0 ||
        ensemble_config_require_summary(ens_config)) {
        // We are expecting there to be summary data
        // The timing information - i.e. mainly what is the last report step
        // in these results are inferred from the loading of summary results,
        // hence we must load the summary results first.
        try {
            auto summary = load_ecl_sum(run_arg_get_runpath(run_arg),
                                        run_arg_get_job_name(run_arg));
            auto status = enkf_state_internalize_dynamic_eclipse_results(
                ens_config, summary, matcher, run_arg_get_sim_fs(run_arg),
                run_arg_get_iens(run_arg));
            ecl_sum_free(summary);
            if (status.first != LOAD_SUCCESSFUL) {
                return {status.first,
                        status.second +
                            fmt::format(" from: {}/{}.UNSMRY",
                                        run_arg_get_runpath(run_arg),
                                        run_arg_get_job_name(run_arg))};
            }
        } catch (std::invalid_argument const &ex) {
            return {LOAD_FAILURE,
                    ex.what() + fmt::format(" from: {}/{}.UNSMRY",
                                            run_arg_get_runpath(run_arg),
                                            run_arg_get_job_name(run_arg))};
        }
    }

    enkf_fs_type *sim_fs = run_arg_get_sim_fs(run_arg);
    int last_report = time_map_get_last_step(enkf_fs_get_time_map(sim_fs));
    if (last_report < 0)
        last_report = model_config_get_last_history_restart(model_config);
    auto result = enkf_state_internalize_GEN_DATA(ens_config, run_arg,
                                                  model_config, last_report);
    if (result == LOAD_FAILURE)
        return {LOAD_FAILURE, "Failed to internalize GEN_DATA"};
    return {LOAD_SUCCESSFUL, "Results loaded successfully."};
}

std::pair<fw_load_status, std::string>
enkf_state_load_from_forward_model(ensemble_config_type *ens_config,
                                   model_config_type *model_config,
                                   const run_arg_type *run_arg) {
    std::pair<fw_load_status, std::string> result;
    if (ensemble_config_have_forward_init(ens_config))
        result = ensemble_config_forward_init(ens_config, run_arg);
    if (result.first == LOAD_SUCCESSFUL) {
        result =
            enkf_state_internalize_results(ens_config, model_config, run_arg);
    }
    auto &state_map = enkf_fs_get_state_map(run_arg_get_sim_fs(run_arg));
    int iens = run_arg_get_iens(run_arg);
    if (result.first != LOAD_SUCCESSFUL)
        state_map.set(iens, STATE_LOAD_FAILURE);
    else
        state_map.set(iens, STATE_HAS_DATA);

    return result;
}

bool enkf_state_complete_forward_model_EXIT_handler__(run_arg_type *run_arg) {
    if (run_arg_get_run_status(run_arg) != JOB_LOAD_FAILURE)
        run_arg_set_run_status(run_arg, JOB_RUN_FAILURE);

    auto &state_map = enkf_fs_get_state_map(run_arg_get_sim_fs(run_arg));
    state_map.set(run_arg_get_iens(run_arg), STATE_LOAD_FAILURE);
    return false;
}

ERT_CLIB_SUBMODULE("enkf_state", m) {
    m.def("state_initialize",
          [](Cwrap<enkf_node_type> param_node, Cwrap<enkf_fs_type> fs,
             int iens) { return enkf_state_initialize(fs, param_node, iens); });

    m.def("internalize_results", [](Cwrap<ensemble_config_type> ens_config,
                                    Cwrap<model_config_type> model_config,
                                    Cwrap<run_arg_type> run_arg) {
        return enkf_state_internalize_results(ens_config, model_config,
                                              run_arg);
    });
}
