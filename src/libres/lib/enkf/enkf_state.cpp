/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_state.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <vector>

#include <ert/python.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.h>
#include <ert/util/rng.h>

#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_sum.h>

#include <ert/job_queue/environment_varlist.hpp>
#include <ert/job_queue/forward_model.hpp>

#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/gen_data.hpp>
#include <ert/logging.hpp>
#include <ert/res_util/memory.hpp>

static auto logger = ert::get_logger("enkf");
#define ENKF_STATE_TYPE_ID 78132

/**
   This struct contains various objects which the enkf_state needs
   during operation, which the enkf_state_object *DOES NOT* own. The
   struct only contains pointers to objects owned by (typically) the
   enkf_main object.

   If the enkf_state object writes to any of the objects in this
   struct that can be considered a serious *BUG*.

   The elements in this struct should not change during the
   application lifetime?
*/
typedef struct shared_info_struct {
    model_config_type *model_config;
    /** The list of external jobs which are installed - and *how* they should
     * be run (with Python code) */
    ext_joblist_type *joblist;
    const site_config_type *site_config;
    ert_templates_type *templates;
    const ecl_config_type *ecl_config;
} shared_info_type;

struct enkf_state_struct {
    UTIL_TYPE_ID_DECLARATION;
    hash_type *node_hash;
    /** The config nodes for the enkf_node objects contained in node_hash. */
    ensemble_config_type *ensemble_config;
    /** Pointers to shared objects which is needed by the enkf_state object (read only). */
    shared_info_type *shared_info;
    int __iens;
};

static shared_info_type *shared_info_alloc(const site_config_type *site_config,
                                           model_config_type *model_config,
                                           const ecl_config_type *ecl_config,
                                           ert_templates_type *templates) {
    shared_info_type *shared_info =
        (shared_info_type *)util_malloc(sizeof *shared_info);
    shared_info->joblist = site_config_get_installed_jobs(site_config);
    shared_info->site_config = site_config;
    shared_info->model_config = model_config;
    shared_info->templates = templates;
    shared_info->ecl_config = ecl_config;
    return shared_info;
}

static void shared_info_free(shared_info_type *shared_info) {
    // Adding something here is a BUG - this object does not own anything.
    free(shared_info);
}

/**
  This function does not acces the nodes of the enkf_state object.
*/
void enkf_state_initialize(enkf_state_type *enkf_state, rng_type *rng,
                           enkf_fs_type *fs,
                           const std::vector<std::string> &param_list,
                           init_mode_type init_mode) {
    int iens = enkf_state->__iens;
    state_map_type *state_map = enkf_fs_get_state_map(fs);
    realisation_state_enum current_state = state_map_iget(state_map, iens);
    if ((current_state == STATE_PARENT_FAILURE) && (init_mode != INIT_FORCE))
        return;
    else {
        const ensemble_config_type *ensemble_config =
            enkf_state->ensemble_config;
        for (auto &param : param_list) {
            const enkf_config_node_type *config_node =
                ensemble_config_get_node(ensemble_config, param.c_str());
            enkf_node_type *param_node = enkf_node_alloc(config_node);
            node_id_type node_id = {.report_step = 0, .iens = iens};
            bool has_data = enkf_node_has_data(param_node, fs, node_id);

            if ((init_mode == INIT_FORCE) || (has_data == false) ||
                (current_state == STATE_LOAD_FAILURE)) {
                if (enkf_node_initialize(param_node, iens, rng))
                    enkf_node_store(param_node, fs, node_id);
            }

            enkf_node_free(param_node);
        }
        state_map_update_matching(state_map, iens,
                                  STATE_UNDEFINED | STATE_LOAD_FAILURE,
                                  STATE_INITIALIZED);
        enkf_fs_fsync(fs);
    }
}

static void enkf_state_add_nodes(enkf_state_type *enkf_state,
                                 const ensemble_config_type *ensemble_config) {
    stringlist_type *container_keys = stringlist_alloc_new();
    stringlist_type *keylist = ensemble_config_alloc_keylist(ensemble_config);
    int keys = stringlist_get_size(keylist);

    // 1: Add all regular nodes
    for (int ik = 0; ik < keys; ik++) {
        const char *key = stringlist_iget(keylist, ik);
        const enkf_config_node_type *config_node =
            ensemble_config_get_node(ensemble_config, key);
        if (enkf_config_node_get_impl_type(config_node) == CONTAINER) {
            stringlist_append_copy(container_keys, key);
        } else
            enkf_state_add_node(enkf_state, key, config_node);
    }

    // 2: Add container nodes - must ensure that all other nodes have
    //    been added already (this implies that containers of containers
    //    will be victim of hash retrieval order problems ....
    for (int ik = 0; ik < stringlist_get_size(container_keys); ik++) {
        const char *key = stringlist_iget(container_keys, ik);
        const enkf_config_node_type *config_node =
            ensemble_config_get_node(ensemble_config, key);
        enkf_state_add_node(enkf_state, key, config_node);
    }

    stringlist_free(keylist);
    stringlist_free(container_keys);
}

ecl_sum_type *load_ecl_sum(const ecl_config_type *ecl_config,
                           const char *run_path, const char *eclbase) {
    ecl_sum_type *summary = NULL;

    const bool fmt_file = ecl_config_get_formatted(ecl_config);
    char *header_file = ecl_util_alloc_exfilename(
        run_path, eclbase, ECL_SUMMARY_HEADER_FILE, fmt_file, -1);
    char *unified_file = ecl_util_alloc_exfilename(
        run_path, eclbase, ECL_UNIFIED_SUMMARY_FILE, fmt_file, -1);
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

enkf_state_type *enkf_state_alloc(int iens, rng_type *rng,
                                  model_config_type *model_config,
                                  ensemble_config_type *ensemble_config,
                                  const site_config_type *site_config,
                                  const ecl_config_type *ecl_config,
                                  ert_templates_type *templates) {

    enkf_state_type *enkf_state =
        (enkf_state_type *)util_malloc(sizeof *enkf_state);
    UTIL_TYPE_ID_INIT(enkf_state, ENKF_STATE_TYPE_ID);

    enkf_state->ensemble_config = ensemble_config;
    enkf_state->shared_info =
        shared_info_alloc(site_config, model_config, ecl_config, templates);
    enkf_state->node_hash = hash_alloc();

    enkf_state->__iens = iens;
    enkf_state_add_nodes(enkf_state, ensemble_config);

    return enkf_state;
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

        if (enkf_node_forward_load(node, report_step, run_arg, NULL)) {
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
static std::pair<fw_load_status, std::string> enkf_state_internalize_results(
    ensemble_config_type *ens_config, model_config_type *model_config,
    const ecl_config_type *ecl_config, const run_arg_type *run_arg) {
    const summary_key_matcher_type *matcher =
        ensemble_config_get_summary_key_matcher(ens_config);

    if (summary_key_matcher_get_size(matcher) > 0 ||
        ensemble_config_require_summary(ens_config)) {
        // We are expecting there to be summary data
        // The timing information - i.e. mainly what is the last report step
        // in these results are inferred from the loading of summary results,
        // hence we must load the summary results first.
        try {
            auto summary =
                load_ecl_sum(ecl_config, run_arg_get_runpath(run_arg),
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
    return {LOAD_SUCCESSFUL, ""};
}

static std::pair<fw_load_status, std::string>
enkf_state_load_from_forward_model__(ensemble_config_type *ens_config,
                                     model_config_type *model_config,
                                     const ecl_config_type *ecl_config,
                                     const run_arg_type *run_arg) {
    std::pair<fw_load_status, std::string> result;
    if (ensemble_config_have_forward_init(ens_config))
        result = ensemble_config_forward_init(ens_config, run_arg);
    if (result.first == LOAD_SUCCESSFUL) {
        result = enkf_state_internalize_results(ens_config, model_config,
                                                ecl_config, run_arg);
    }
    state_map_type *state_map =
        enkf_fs_get_state_map(run_arg_get_sim_fs(run_arg));
    int iens = run_arg_get_iens(run_arg);
    if (result.first != LOAD_SUCCESSFUL)
        state_map_iset(state_map, iens, STATE_LOAD_FAILURE);
    else
        state_map_iset(state_map, iens, STATE_HAS_DATA);

    return result;
}

std::pair<fw_load_status, std::string>
enkf_state_load_from_forward_model(enkf_state_type *enkf_state,
                                   run_arg_type *run_arg) {

    ensemble_config_type *ens_config = enkf_state->ensemble_config;
    model_config_type *model_config = enkf_state->shared_info->model_config;
    const ecl_config_type *ecl_config = enkf_state->shared_info->ecl_config;

    return enkf_state_load_from_forward_model__(ens_config, model_config,
                                                ecl_config, run_arg);
}

void enkf_state_free(enkf_state_type *enkf_state) {
    hash_free(enkf_state->node_hash);
    shared_info_free(enkf_state->shared_info);
    free(enkf_state);
}

std::pair<fw_load_status, std::string>
enkf_state_complete_forward_modelOK(const res_config_type *res_config,
                                    run_arg_type *run_arg) {

    ensemble_config_type *ens_config =
        res_config_get_ensemble_config(res_config);
    const ecl_config_type *ecl_config = res_config_get_ecl_config(res_config);
    model_config_type *model_config = res_config_get_model_config(res_config);
    auto result = enkf_state_load_from_forward_model__(ens_config, model_config,
                                                       ecl_config, run_arg);

    if (result.first == LOAD_SUCCESSFUL) {
        result.second = "Results loaded successfully.";
    }

    return result;
}

bool enkf_state_complete_forward_model_EXIT_handler__(run_arg_type *run_arg) {
    if (run_arg_get_run_status(run_arg) != JOB_LOAD_FAILURE)
        run_arg_set_run_status(run_arg, JOB_RUN_FAILURE);

    state_map_type *state_map =
        enkf_fs_get_state_map(run_arg_get_sim_fs(run_arg));
    state_map_iset(state_map, run_arg_get_iens(run_arg), STATE_LOAD_FAILURE);
    return false;
}

#include "enkf_state_nodes.cpp"

RES_LIB_SUBMODULE("enkf_state", m) {
    m.def("state_initialize", [](py::object enkf_main, py::object fs,
                                 std::vector<std::string> &param_list,
                                 int init_mode, int iens) {
        auto enkf_main_ = ert::from_cwrap<enkf_main_type>(enkf_main);
        auto fs_ = ert::from_cwrap<enkf_fs_type>(fs);
        init_mode_type init_mode_ = static_cast<init_mode_type>(init_mode);
        return enkf_state_initialize(
            enkf_main_iget_state(enkf_main_, iens),
            rng_manager_iget(enkf_main_get_rng_manager(enkf_main_), iens), fs_,
            param_list, init_mode_);
    });
}
