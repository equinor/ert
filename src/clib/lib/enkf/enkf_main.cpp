/*
   Copyright (C) 2011  Equinor ASA, Norway.
   The file 'enkf_main.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <future>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <ert/python.hpp>
#include <ert/res_util/path_fmt.hpp>
#include <ert/util/bool_vector.h>
#include <ert/util/hash.h>
#include <ert/util/int_vector.h>
#include <ert/util/rng.h>
#include <ert/util/type_vector_functions.h>
#include <ert/util/vector.hpp>

#include <ert/logging.hpp>
#include <ert/res_util/subst_list.hpp>

#include <ert/sched/history.hpp>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/enkf_linalg.hpp>
#include <ert/analysis/update.hpp>

#include <ert/enkf/enkf_analysis.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_main.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/field.hpp>
#include <ert/enkf/obs_data.hpp>

#include <ert/concurrency.hpp>

#include <ert/python.hpp>

static auto logger = ert::get_logger("enkf");

namespace fs = std::filesystem;

#define ENKF_MAIN_ID 8301

struct enkf_state_deleter {
    void operator()(enkf_state_type *p) const { enkf_state_free(p); };
};
using enkf_state = std::shared_ptr<enkf_state_type>;

/**
   This object should contain **everything** needed to run a enkf
   simulation. A way to wrap up all available information/state and
   pass it around. An attempt has been made to collect various pieces
   of related information together in a couple of objects
   (model_config, ecl_config, site_config and ensemble_config). When
   it comes to these holding objects the following should be observed:

    1. It not always obvious where a piece of information should be
       stored, i.e. the grid is a property of the model, however it is
       an eclipse grid, and hence also belongs to eclipse
       configuration?? [In this case ecl_config wins out.]

    2. The information stored in these objects is typically passed on
       to the enkf_state object, where it is used.

    3. At enkf_state level it is not really consequent - in some cases
       the enkf_state object takes a scalar copy ,
       and in other cases only a pointer down to the underlying
       enkf_main object is taken. In the former case it is no way to
       change global behaviour by modifying the enkf_main objects.

       In the enkf_state object the fields of the member_config,
       ecl_config, site_config and ensemble_config objects are mixed
       and matched into other small holding objects defined in
       enkf_state.c.
*/
struct enkf_main_struct {
    UTIL_TYPE_ID_DECLARATION;
    /** The internalized information. */
    enkf_fs_type *dbase;

    const res_config_type *res_config;
    rng_manager_type *rng_manager;
    rng_type *shared_rng;

    enkf_obs_type *obs;

    /** The ensemble */
    std::vector<enkf_state> ensemble;
    /** The size of the ensemble */
    int ens_size;
};

UTIL_SAFE_CAST_FUNCTION(enkf_main, ENKF_MAIN_ID)
UTIL_IS_INSTANCE_FUNCTION(enkf_main, ENKF_MAIN_ID)

const res_config_type *
enkf_main_get_res_config(const enkf_main_type *enkf_main) {
    return enkf_main->res_config;
}

subst_list_type *enkf_main_get_data_kw(const enkf_main_type *enkf_main) {
    return subst_config_get_subst_list(
        res_config_get_subst_config(enkf_main_get_res_config(enkf_main)));
}

enkf_obs_type *enkf_main_get_obs(const enkf_main_type *enkf_main) {
    return enkf_main->obs;
}

bool enkf_main_have_obs(const enkf_main_type *enkf_main) {
    return enkf_obs_have_obs(enkf_main->obs);
}

const hook_manager_type *
enkf_main_get_hook_manager(const enkf_main_type *enkf_main) {
    return res_config_get_hook_manager(enkf_main->res_config);
}

bool enkf_main_load_obs(enkf_main_type *enkf_main, const char *obs_config_file,
                        bool clear_existing) {
    if (clear_existing)
        enkf_obs_clear(enkf_main->obs);

    if (!enkf_obs_is_valid(enkf_main->obs)) {
        fprintf(stderr,
                "** Warning: failed to load observation data from: %s \n",
                obs_config_file);
        return false;
    }

    enkf_obs_load(enkf_main->obs, obs_config_file,
                  analysis_config_get_std_cutoff(
                      res_config_get_analysis_config(enkf_main->res_config)));
    return true;
}

void enkf_main_free(enkf_main_type *enkf_main) {
    if (enkf_main->rng_manager)
        rng_manager_free(enkf_main->rng_manager);

    if (enkf_main->shared_rng)
        rng_free(enkf_main->shared_rng);

    if (enkf_main->obs)
        enkf_obs_free(enkf_main->obs);

    if (enkf_main->dbase != NULL)
        enkf_fs_decref(enkf_main->dbase);

    delete enkf_main;
}

rng_type *enkf_main_get_shared_rng(enkf_main_type *enkf_main) {
    return enkf_main->shared_rng;
}

int enkf_main_get_ensemble_size(const enkf_main_type *enkf_main) {
    return enkf_main->ens_size;
}

void enkf_main_get_observations(const enkf_main_type *enkf_main,
                                const char *user_key, int obs_count,
                                time_t *obs_time, double *y, double *std) {
    ensemble_config_get_observations(
        res_config_get_ensemble_config(enkf_main->res_config), enkf_main->obs,
        user_key, obs_count, obs_time, y, std);
}

void enkf_main_install_SIGNALS(void) { util_install_signals(); }

ert_workflow_list_type *enkf_main_get_workflow_list(enkf_main_type *enkf_main) {
    return res_config_get_workflow_list(enkf_main->res_config);
}

int enkf_main_load_from_run_context(enkf_main_type *enkf_main,
                                    std::vector<bool> active_mask,
                                    enkf_fs_type *sim_fs,
                                    std::vector<run_arg_type *> run_args) {
    auto const ens_size = enkf_main_get_ensemble_size(enkf_main);

    // Loading state from a fwd-model is mainly io-bound so we can
    // allow a lot more than #cores threads to execute in parallel.
    // The number 100 is quite arbitrarily chosen though and should
    // probably come from some resource like a site-config or similar.
    // NOTE that this mechanism only limits the number of *concurrently
    // executing* threads. The number of instantiated and stored futures
    // will be equal to the number of active realizations.
    Semafoor concurrently_executing_threads(100);
    std::vector<
        std::tuple<int, std::future<std::pair<fw_load_status, std::string>>>>
        futures;

    // If this function is called via pybind11 we need to release
    // the GIL here because this function may spin up several
    // threads which also may need the GIL (e.g. for logging)
    PyThreadState *state = nullptr;
    if (PyGILState_Check() == 1)
        state = PyEval_SaveThread();

    for (int iens = 0; iens < ens_size; ++iens) {
        if (active_mask[iens]) {

            futures.push_back(std::make_tuple(
                iens, // for logging later
                std::async(
                    std::launch::async,
                    [=](const int realisation, Semafoor &execution_limiter) {
                        // Acquire permit from semaphore or pause execution
                        // until one becomes available. A successfully acquired
                        // permit is released when exiting scope.
                        std::scoped_lock lock(execution_limiter);

                        auto &state_map = enkf_fs_get_state_map(sim_fs);

                        state_map.update_undefined(realisation,
                                                   STATE_INITIALIZED);
                        auto status = enkf_state_load_from_forward_model(
                            enkf_main_iget_state(enkf_main, realisation),
                            run_args[iens]);
                        state_map.set(realisation,
                                      status.first == LOAD_SUCCESSFUL
                                          ? STATE_HAS_DATA
                                          : STATE_LOAD_FAILURE);
                        return status;
                    },
                    iens, std::ref(concurrently_executing_threads))));
        }
    }

    int loaded = 0;
    for (auto &[iens, fut] : futures) {
        auto result = fut.get();
        if (result.first == LOAD_SUCCESSFUL) {
            loaded++;
        } else {
            logger->error("Realization: {}, load failure: {}", iens,
                          result.second);
        }
    }
    if (state)
        PyEval_RestoreThread(state);

    return loaded;
}

rng_manager_type *enkf_main_get_rng_manager(const enkf_main_type *enkf_main) {
    return enkf_main->rng_manager;
}

std::vector<std::string> get_observation_keys(py::object self) {
    auto enkf_main = ert::from_cwrap<enkf_main_type>(self);
    std::vector<std::string> observations;

    hash_iter_type *obs_iter = enkf_obs_alloc_iter(enkf_main->obs);
    while (!hash_iter_is_complete(obs_iter)) {
        const char *obs_key = hash_iter_get_next_key(obs_iter);
        observations.push_back(obs_key);
    }
    hash_iter_free(obs_iter);
    return observations;
}

std::vector<std::string> get_parameter_keys(py::object self) {
    auto enkf_main = ert::from_cwrap<enkf_main_type>(self);

    std::vector<std::string> parameters;
    std::vector<std::string> keylist = ensemble_config_keylist_from_var_type(
        res_config_get_ensemble_config(enkf_main->res_config), PARAMETER);

    // Add all GEN_KW keywords to parameters that is not
    // the SCHEDULE_PREDICTION_FILE
    std::copy_if(keylist.begin(), keylist.end(), std::back_inserter(parameters),
                 [](auto key) { return key != "PRED"; });

    return parameters;
}

namespace enkf_main {
/** @brief Writes the eclipse data file
 *
 *  Substitutes the parameters of the templated ECL_DATA_FILE
 *  and writes it to the runpath.
 *
 * @param data_file_template The template for the data file.
 * @param run_arg Contains the information about the given run.
 */
void write_eclipse_data_file(const char *data_file_template,
                             const char *run_path, const char *job_name,
                             const subst_list_type *subst_list) {
    char *data_file_destination =
        ecl_util_alloc_filename(run_path, job_name, ECL_DATA_FILE, true, -1);

    //Perform substitutions on the data file destination path
    subst_list_update_string(subst_list, &data_file_destination);

    //Perform substitutions on the data file template contents
    subst_list_filter_file(subst_list, data_file_template,
                           data_file_destination);

    free(data_file_destination);
}

/**
  @brief Substitutes the sampled parameters into the runpath.

  Handles the substitution of all sampled parameter values into parameter
  templates. E.g. for configs including `GEN_KW key template_file target_file`,
  sampled values are gotten from fs, replaced in the contents of template_file
  and written to target_file in the runpath.

  @param ens_config Where to find the nodes (e.g. `GEN_KW key template_file
    target_file` definition).
  @param export_base_name The base name of the value_export file (e.g. if
    "parameters", value export file will e.g. be "parameters.json")
  @param run_path The run__path to write the target file in.
  @param iens The realization number.
  @param fs The enkf_fs to load sampled parameters from
*/
void ecl_write(const ensemble_config_type *ens_config,
               const char *export_base_name, const char *run_path, int iens,
               enkf_fs_type *fs) {
    value_export_type *export_value =
        value_export_alloc(run_path, export_base_name);

    for (auto &key : ensemble_config_keylist_from_var_type(
             ens_config, PARAMETER + EXT_PARAMETER)) {
        enkf_node_type *enkf_node =
            enkf_node_alloc(ensemble_config_get_node(ens_config, key.c_str()));
        node_id_type node_id = {.report_step = 0, .iens = iens};

        if (enkf_node_use_forward_init(enkf_node) &&
            !enkf_node_has_data(enkf_node, fs, node_id))
            continue;
        enkf_node_load(enkf_node, fs, node_id);

        enkf_node_ecl_write(enkf_node, run_path, export_value, 0);
        enkf_node_free(enkf_node);
    }
    value_export(export_value);

    value_export_free(export_value);
}

/**
 * @brief Initializes an active run.
 *
 *  * Instantiate res_config_templates which substitutes arg_list from the template
 *      and from run_arg into each template and writes it to runpath;
 *  * substitutes sampled parameters into the parameter nodes and write to runpath;
 *  * substitutes DATAKW into the eclipse data file template and write it to runpath;
 *  * write the job script.
 *
 * @param res_config The config to use for initialization.
 * @param run_path The runpath string
 * @param iens The realization number.
 * @param fs The file system to write to
 * @param run_id Unique id of run
 * @param subst_list The substitutions to perform for that run.
 */
void init_active_run(const res_config_type *res_config, char *run_path,
                     int iens, enkf_fs_type *fs, char *run_id, char *job_name,
                     const subst_list_type *subst_list) {

    model_config_type *model_config = res_config_get_model_config(res_config);
    ensemble_config_type *ens_config =
        res_config_get_ensemble_config(res_config);

    ecl_write(ens_config, model_config_get_gen_kw_export_name(model_config),
              run_path, iens, fs);

    // Create the eclipse data file (if eclbase and DATA_FILE)
    const ecl_config_type *ecl_config = res_config_get_ecl_config(res_config);
    const char *data_file_template = ecl_config_get_data_file(ecl_config);
    if (ecl_config_have_eclbase(ecl_config) && data_file_template) {
        write_eclipse_data_file(data_file_template, run_path, job_name,
                                subst_list);
    }

    // Create the job script
    const site_config_type *site_config =
        res_config_get_site_config(res_config);
    forward_model_formatted_fprintf(
        model_config_get_forward_model(model_config), run_id, run_path,
        model_config_get_data_root(model_config), subst_list,
        site_config_get_umask(site_config),
        site_config_get_env_varlist(site_config));
}
} // namespace enkf_main

enkf_state_type *enkf_main_iget_state(const enkf_main_type *enkf_main,
                                      int iens) {
    return enkf_main->ensemble.at(iens).get();
}

bool enkf_main_case_is_current(const enkf_main_type *enkf_main,
                               const char *case_path) {
    char *mount_point = enkf_main_alloc_mount_point(enkf_main, case_path);
    const char *current_mount_point = NULL;
    bool is_current;

    if (enkf_main->dbase != NULL)
        current_mount_point = enkf_fs_get_mount_point(enkf_main->dbase);

    is_current = util_string_equal(mount_point, current_mount_point);
    free(mount_point);
    return is_current;
}

static bool
enkf_main_current_case_file_exists(const enkf_main_type *enkf_main) {
    const char *ens_path = model_config_get_enspath(
        res_config_get_model_config(enkf_main->res_config));
    char *current_case_file =
        util_alloc_filename(ens_path, CURRENT_CASE_FILE, NULL);
    bool exists = fs::exists(current_case_file);
    free(current_case_file);
    return exists;
}

char *enkf_main_read_alloc_current_case_name(const enkf_main_type *enkf_main) {
    char *current_case = NULL;
    const char *ens_path = model_config_get_enspath(
        res_config_get_model_config(enkf_main->res_config));
    char *current_case_file =
        util_alloc_filename(ens_path, CURRENT_CASE_FILE, NULL);
    if (enkf_main_current_case_file_exists(enkf_main)) {
        FILE *stream = util_fopen(current_case_file, "r");
        current_case = util_fscanf_alloc_token(stream);
        fclose(stream);
    } else {
        util_abort("%s: File: storage/current_case not found, aborting! \n",
                   __func__);
    }
    free(current_case_file);
    return current_case;
}

static void enkf_main_copy_ensemble(const ensemble_config_type *ensemble_config,
                                    enkf_fs_type *source_case_fs,
                                    int source_report_step,
                                    enkf_fs_type *target_case_fs,
                                    const std::vector<bool> &iens_mask,
                                    const std::vector<std::string> &node_list) {
    auto &target_state_map = enkf_fs_get_state_map(target_case_fs);

    for (auto &node : node_list) {
        enkf_config_node_type *config_node =
            ensemble_config_get_node(ensemble_config, node.c_str());

        int src_iens = 0;
        for (auto mask : iens_mask) {
            if (mask) {
                node_id_type src_id = {.report_step = source_report_step,
                                       .iens = src_iens};
                node_id_type target_id = {.report_step = 0, .iens = src_iens};

                /* The copy is careful ... */
                if (enkf_config_node_has_node(config_node, source_case_fs,
                                              src_id))
                    enkf_node_copy(config_node, source_case_fs, target_case_fs,
                                   src_id, target_id);

                target_state_map.set(src_iens, STATE_INITIALIZED);
            }
            src_iens++;
        }
    }
}

void enkf_main_init_current_case_from_existing(enkf_main_type *enkf_main,
                                               enkf_fs_type *source_case_fs,
                                               int source_report_step) {

    enkf_fs_type *current_fs = enkf_main_get_fs(enkf_main);

    enkf_main_init_case_from_existing(enkf_main, source_case_fs,
                                      source_report_step, current_fs);
}

static void enkf_main_init_current_case_from_existing_custom(
    enkf_main_type *enkf_main, enkf_fs_type *source_case_fs,
    int source_report_step, std::vector<std::string> &node_list,
    std::vector<bool> &iactive) {

    enkf_fs_type *current_fs = enkf_main_get_fs(enkf_main);

    enkf_main_copy_ensemble(
        res_config_get_ensemble_config(enkf_main->res_config), source_case_fs,
        source_report_step, current_fs, iactive, node_list);
    enkf_fs_fsync(current_fs);
}

void enkf_main_init_case_from_existing(const enkf_main_type *enkf_main,
                                       enkf_fs_type *source_case_fs,
                                       int source_report_step,
                                       enkf_fs_type *target_case_fs) {

    std::vector<std::string> param_list = ensemble_config_keylist_from_var_type(
        res_config_get_ensemble_config(enkf_main->res_config),
        PARAMETER); /* Select only paramters - will fail for GEN_DATA of type DYNAMIC_STATE. */
    std::vector<bool> iactive(enkf_main_get_ensemble_size(enkf_main), true);
    enkf_main_copy_ensemble(
        res_config_get_ensemble_config(enkf_main->res_config), source_case_fs,
        source_report_step, target_case_fs, iactive, param_list);

    enkf_fs_fsync(target_case_fs);
}

/**
   This function will go through the filesystem and check that we have
   initial data for all parameters and all realizations. If the second
   argument mask is different from NULL, the function will only
   consider the realizations for which mask is true (if mask == NULL
   all realizations will be checked).
*/
static bool
enkf_main_case_is_initialized__(const ensemble_config_type *ensemble_config,
                                enkf_fs_type *fs, const int ens_size) {
    std::vector<std::string> parameter_keys =
        ensemble_config_keylist_from_var_type(ensemble_config, PARAMETER);
    bool initialized = true;
    for (int ikey = 0; (ikey < parameter_keys.size()) && initialized; ikey++) {
        const enkf_config_node_type *config_node = ensemble_config_get_node(
            ensemble_config, parameter_keys[ikey].c_str());
        initialized = enkf_config_node_has_node(config_node, fs,
                                                {.report_step = 0, .iens = 0});
        for (int iens = 0; (iens < ens_size) && initialized; iens++) {
            initialized = enkf_config_node_has_node(
                config_node, fs, {.report_step = 0, .iens = iens});
        }
    }

    return initialized;
}

bool enkf_main_case_is_initialized(const enkf_main_type *enkf_main,
                                   const char *case_name) {
    enkf_fs_type *fs = enkf_main_mount_alt_fs(enkf_main, case_name, false);
    if (fs) {
        bool initialized = enkf_main_case_is_initialized__(
            res_config_get_ensemble_config(enkf_main->res_config), fs,
            enkf_main->ens_size);
        enkf_fs_decref(fs);
        return initialized;
    } else
        return false;
}

static void update_case_log(enkf_main_type *enkf_main, const char *case_path) {
    /*  : Update a small text file with the name of the host currently
        running ert, the pid number of the process, the active case
        and when it started.

        If the previous shutdown was unclean the file will be around,
        and we will need the info from the previous invocation which
        is in the file. For that reason we open with mode 'a' instead
        of 'w'.
  */

    const char *ens_path = model_config_get_enspath(
        res_config_get_model_config(enkf_main->res_config));

    {
        int buffer_size = 256;
        char *current_host = util_alloc_filename(ens_path, CASE_LOG, NULL);
        FILE *stream = util_fopen(current_host, "a");

        fprintf(stream, "CASE:%-16s  ", case_path);
        fprintf(stream, "PID:%-8d  ", getpid());
        {
            char hostname[buffer_size];
            gethostname(hostname, buffer_size);
            fprintf(stream, "HOST:%-16s  ", hostname);
        }

        {
            int year, month, day, hour, minute, second;
            time_t now = time(NULL);

            util_set_datetime_values_utc(now, &second, &minute, &hour, &day,
                                         &month, &year);

            fprintf(stream, "TIME:%04d-%02d-%02dT%02d:%02d:%02d\n", year, month,
                    day, hour, minute, second);
        }
        fclose(stream);
        free(current_host);
    }
}

static void enkf_main_write_current_case_file(const enkf_main_type *enkf_main,
                                              const char *case_path) {
    const char *ens_path = model_config_get_enspath(
        res_config_get_model_config(enkf_main->res_config));
    const char *base = CURRENT_CASE_FILE;
    char *current_case_file = util_alloc_filename(ens_path, base, NULL);
    FILE *stream = util_fopen(current_case_file, "w");
    fprintf(stream, "%s", case_path);
    fclose(stream);
    free(current_case_file);
}

static void enkf_main_gen_data_special(enkf_main_type *enkf_main,
                                       enkf_fs_type *fs) {
    ensemble_config_type *ensemble_config =
        res_config_get_ensemble_config(enkf_main->res_config);
    stringlist_type *gen_data_keys =
        ensemble_config_alloc_keylist_from_impl_type(ensemble_config, GEN_DATA);
    for (int i = 0; i < stringlist_get_size(gen_data_keys); i++) {
        enkf_config_node_type *config_node = ensemble_config_get_node(
            ensemble_config, stringlist_iget(gen_data_keys, i));
        gen_data_config_type *gen_data_config =
            (gen_data_config_type *)enkf_config_node_get_ref(config_node);

        if (gen_data_config_is_dynamic(gen_data_config))
            gen_data_config_set_ens_size(gen_data_config, enkf_main->ens_size);
    }
    stringlist_free(gen_data_keys);
}

static void
enkf_main_update_current_case(enkf_main_type *enkf_main,
                              const char *case_path /* Can be NULL */) {
    if (!case_path)
        case_path = enkf_fs_get_case_name(enkf_main_get_fs(enkf_main));

    enkf_main_write_current_case_file(enkf_main, case_path);
    update_case_log(enkf_main, case_path);

    enkf_main_gen_data_special(enkf_main, enkf_main_get_fs(enkf_main));
}

static void enkf_main_create_fs(const enkf_main_type *enkf_main,
                                const char *case_path) {
    char *new_mount_point = enkf_main_alloc_mount_point(enkf_main, case_path);

    enkf_fs_create_fs(new_mount_point, BLOCK_FS_DRIVER_ID, false);

    free(new_mount_point);
}

char *enkf_main_alloc_mount_point(const enkf_main_type *enkf_main,
                                  const char *case_path) {
    char *mount_point;
    if (util_is_abs_path(case_path))
        mount_point = util_alloc_string_copy(case_path);
    else
        mount_point = util_alloc_filename(
            model_config_get_enspath(
                res_config_get_model_config(enkf_main->res_config)),
            case_path, NULL);
    return mount_point;
}

/**
  Return a weak reference - i.e. the refcount is not increased.
*/
enkf_fs_type *enkf_main_get_fs(const enkf_main_type *enkf_main) {
    return enkf_main->dbase;
}

enkf_fs_type *enkf_main_get_fs_ref(const enkf_main_type *enkf_main) {
    return enkf_fs_get_ref(enkf_main->dbase);
}

/**
  This function will return a valid enkf_fs instance; either just a
  pointer to the current enkf_main->dbase, or alternatively it will
  create a brand new fs instance. Because we do not really know whether
  a new instance has been created or not resource handling becomes
  slightly non trivial:


    1. When calling scope is finished with the enkf_fs instance it
       must call enkf_fs_decref(); the enkf_fs_decref() function will
       close the filesystem and free all resources when the reference
       count has reached zero.
*/
enkf_fs_type *enkf_main_mount_alt_fs(const enkf_main_type *enkf_main,
                                     const char *case_path, bool create,
                                     bool read_only) {
    if (enkf_main_case_is_current(enkf_main, case_path)) {
        // Fast path - we just return a reference to the currently selected case;
        // with increased refcount.
        enkf_fs_incref(enkf_main->dbase);
        return enkf_main->dbase;
    } else {
        // We have asked for an alterantive fs - must mount and possibly create that first.
        enkf_fs_type *new_fs = NULL;
        if (case_path != NULL) {
            char *new_mount_point =
                enkf_main_alloc_mount_point(enkf_main, case_path);

            if (!enkf_fs_exists(new_mount_point)) {
                if (create)
                    enkf_main_create_fs(enkf_main, case_path);
            }

            new_fs = enkf_fs_mount(new_mount_point, read_only);
            if (new_fs) {
                const model_config_type *model_config =
                    res_config_get_model_config(enkf_main->res_config);
                const ecl_sum_type *refcase =
                    model_config_get_refcase(model_config);

                if (refcase) {
                    time_map_type *time_map = enkf_fs_get_time_map(new_fs);
                    if (!time_map_attach_refcase(time_map, refcase))
                        logger->error("Warning mismatch between refcase:{} "
                                      "and existing case:{}",
                                      ecl_sum_get_case(refcase),
                                      new_mount_point);
                }
            }

            free(new_mount_point);
        }
        return new_fs;
    }
}

static void enkf_main_update_summary_config_from_fs__(enkf_main_type *enkf_main,
                                                      enkf_fs_type *fs) {
    ensemble_config_type *ensemble_config =
        res_config_get_ensemble_config(enkf_main->res_config);
    summary_key_set_type *summary_key_set = enkf_fs_get_summary_key_set(fs);
    stringlist_type *keys = summary_key_set_alloc_keys(summary_key_set);

    for (int i = 0; i < stringlist_get_size(keys); i++) {
        const char *key = stringlist_iget(keys, i);
        ensemble_config_add_summary(ensemble_config, key, LOAD_FAIL_SILENT);
    }
    stringlist_free(keys);
}

/**
   The enkf_fs instances employ a simple reference counting
   scheme. The main point with this system is to avoid opening the
   full timesystem more than necessary (this is quite compute
   intensive). This is essentially achieved by:

      1. Create new fs instances by using the function
         enkf_main_mount_alt_fs() - depending on the input arguments
         this will either create a new enkf_fs instance or it will
         just return a pointer to currently open fs instance; with an
         increased refcount.

      2. When you are finished with working with filesystem pointer
         call enkf_fs_unmount() - this will reduce the refcount with
         one, and eventually discard the complete datastructure when
         the refcount has reached zero.

      3. By using the function enkf_main_get_fs() /
         enkf_fs_get_weakref() you get a pointer to the current fs
         instance WITHOUT INCREASING THE REFCOUNT. This means that
         scope calling one of these functions does not get any
         ownership to the enkf_fs instance.

   The enkf_main instance will take ownership of the enkf_fs instance;
   this implies that the calling scope must have proper ownership of
   the fs instance which is passed in. The return value from
   enkf_main_get_fs() can NOT be used as input to this function; this
   is not checked for in any way - but the crash will be horrible if
   this is not adhered to.
*/
void enkf_main_set_fs(enkf_main_type *enkf_main, enkf_fs_type *fs,
                      const char *case_path /* Can be NULL */) {
    if (enkf_main->dbase != fs) {
        enkf_fs_incref(fs);

        if (enkf_main->dbase)
            enkf_fs_decref(enkf_main->dbase);

        enkf_main->dbase = fs;
        enkf_main_update_current_case(enkf_main, case_path);

        enkf_main_update_summary_config_from_fs__(enkf_main, fs);
    }
}

void enkf_main_select_fs(enkf_main_type *enkf_main, const char *case_path,
                         bool read_only) {
    if (enkf_main_case_is_current(enkf_main, case_path))
        return; /* We have tried to select the currently selected case - just return. */
    else {
        enkf_fs_type *new_fs =
            enkf_main_mount_alt_fs(enkf_main, case_path, true, read_only);
        if (enkf_main->dbase == new_fs)
            util_abort("%s : return reference to current FS in situation where "
                       "that should not happen.\n",
                       __func__);

        if (new_fs != NULL)
            enkf_main_set_fs(enkf_main, new_fs, case_path);
        else {
            const char *ens_path = model_config_get_enspath(
                res_config_get_model_config(enkf_main->res_config));
            util_exit("%s: select filesystem %s:%s failed \n", __func__,
                      ens_path, case_path);
        }
        enkf_fs_decref(new_fs);
    }
}

StateMap enkf_main_read_state_map(const enkf_main_type *enkf_main,
                                  const char *case_path) {
    char *mount_point = enkf_main_alloc_mount_point(enkf_main, case_path);
    auto state_map = enkf_fs_read_state_map(mount_point);
    free(mount_point);
    return state_map;
}

/**
   This function boots everything needed for running a EnKF
   application from the provided res_config.
*/
enkf_main_type *enkf_main_alloc(const res_config_type *res_config,
                                bool read_only) {
    const ecl_config_type *ecl_config = res_config_get_ecl_config(res_config);
    const model_config_type *model_config =
        res_config_get_model_config(res_config);

    enkf_main_type *enkf_main = new enkf_main_type;
    UTIL_TYPE_ID_INIT(enkf_main, ENKF_MAIN_ID);

    enkf_main->res_config = res_config;

    // Init rng
    enkf_main->rng_manager = rng_config_alloc_rng_manager(
        res_config_get_rng_config(enkf_main->res_config));
    enkf_main->shared_rng = rng_manager_alloc_rng(enkf_main->rng_manager);

    // Init storage
    enkf_main->dbase = NULL;
    const char *ens_path = model_config_get_enspath(
        res_config_get_model_config(enkf_main->res_config));
    if (enkf_main_current_case_file_exists(enkf_main)) {
        char *current_case = enkf_main_read_alloc_current_case_name(enkf_main);
        enkf_main_select_fs(enkf_main, current_case, read_only);
        free(current_case);
    } else
        // Selecting (a new) default case
        enkf_main_select_fs(enkf_main, DEFAULT_CASE, read_only);

    // Init observations
    auto obs = enkf_obs_alloc(model_config_get_history(model_config),
                              model_config_get_external_time_map(model_config),
                              ecl_config_get_grid(ecl_config),
                              ecl_config_get_refcase(ecl_config),
                              res_config_get_ensemble_config(res_config));
    const char *obs_config_file =
        model_config_get_obs_config_file(model_config);
    if (obs_config_file)
        enkf_obs_load(obs, obs_config_file,
                      analysis_config_get_std_cutoff(
                          res_config_get_analysis_config(res_config)));
    enkf_main->obs = obs;

    // Add ensemble
    int num_realizations = model_config_get_num_realizations(model_config);
    std::vector<enkf_state> ensemble;
    for (int iens = 0; iens < num_realizations; iens++)
        // Observe that due to the initialization of the rng - this function is currently NOT thread safe.
        ensemble.emplace_back(
            enkf_state_alloc(iens,
                             rng_manager_iget(enkf_main->rng_manager, iens),
                             res_config_get_model_config(res_config),
                             res_config_get_ensemble_config(res_config),
                             res_config_get_site_config(res_config),
                             res_config_get_ecl_config(res_config)),
            enkf_state_deleter());
    enkf_main->ensemble = ensemble;
    enkf_main->ens_size = num_realizations;

    return enkf_main;
}

ERT_CLIB_SUBMODULE("enkf_main", m) {
    using namespace py::literals;
    m.def(
        "init_current_case_from_existing_custom",
        [](py::object self, py::object source_case_py, int source_report_step,
           std::vector<std::string> &node_list, std::vector<bool> &iactive) {
            auto source_case_fs = ert::from_cwrap<enkf_fs_type>(source_case_py);
            auto enkf_main = ert::from_cwrap<enkf_main_type>(self);
            return enkf_main_init_current_case_from_existing_custom(
                enkf_main, source_case_fs, source_report_step, node_list,
                iactive);
        },
        py::arg("self"), py::arg("source_case"), py::arg("source_report_step"),
        py::arg("node_list"), py::arg("iactive"));
    m.def("get_observation_keys", get_observation_keys);
    m.def("get_parameter_keys", get_parameter_keys);
    m.def("load_from_run_context",
          [](py::object self, std::vector<py::object> run_args_,
             std::vector<bool> active_mask, py::object sim_fs_) {
              auto enkf_main = ert::from_cwrap<enkf_main_type>(self);
              auto sim_fs = ert::from_cwrap<enkf_fs_type>(sim_fs_);
              std::vector<run_arg_type *> run_args;
              for (auto &run_arg : run_args_) {
                  run_args.push_back(ert::from_cwrap<run_arg_type>(run_arg));
              }
              return enkf_main_load_from_run_context(enkf_main, active_mask,
                                                     sim_fs, run_args);
          });
    m.def(
        "init_active_run",
        [](py::object res_config, char *run_path, int iens, py::object sim_fs,
           char *run_id, char *job_name, py::object subst_list) {
            enkf_main::init_active_run(
                ert::from_cwrap<res_config_type>(res_config), run_path, iens,
                ert::from_cwrap<enkf_fs_type>(sim_fs), run_id, job_name,
                ert::from_cwrap<subst_list_type>(subst_list));
        },
        py::arg("res_config"), py::arg("run_path"), py::arg("iens"),
        py::arg("sim_fs"), py::arg("run_id"), py::arg("job_name"),
        py::arg("subst_list"));

    m.def(
        "read_state_map",
        [](py::handle self, const std::string &ensemble_name) {
            auto enkf_main = ert::from_cwrap<enkf_main_type>(self);
            return enkf_main_read_state_map(enkf_main, ensemble_name.c_str());
        },
        "self"_a, "ensemble_name"_a);
}
