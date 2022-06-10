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
#include <ert/enkf/enkf_main.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/field.hpp>
#include <ert/enkf/obs_data.hpp>

#include <ert/concurrency.hpp>

#include <ert/python.hpp>

static auto logger = ert::get_logger("enkf");

#define ENKF_MAIN_ID 8301

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
    enkf_state_type **ensemble;
    /** The size of the ensemble */
    int ens_size;
};

void enkf_main_init_internalization(enkf_main_type *);
static void enkf_main_close_fs(enkf_main_type *enkf_main);
static void enkf_main_user_select_initial_fs(enkf_main_type *enkf_main);
static void enkf_main_free_ensemble(enkf_main_type *enkf_main);

UTIL_SAFE_CAST_FUNCTION(enkf_main, ENKF_MAIN_ID)
UTIL_IS_INSTANCE_FUNCTION(enkf_main, ENKF_MAIN_ID)

const analysis_config_type *
enkf_main_get_analysis_config(const enkf_main_type *enkf_main) {
    return res_config_get_analysis_config(enkf_main->res_config);
}

const char *enkf_main_get_site_config_file(const enkf_main_type *enkf_main) {
    return site_config_get_config_file(enkf_main_get_site_config(enkf_main));
}

ensemble_config_type *
enkf_main_get_ensemble_config(const enkf_main_type *enkf_main) {
    return res_config_get_ensemble_config(enkf_main->res_config);
}

const site_config_type *
enkf_main_get_site_config(const enkf_main_type *enkf_main) {
    return res_config_get_site_config(enkf_main->res_config);
}

const res_config_type *
enkf_main_get_res_config(const enkf_main_type *enkf_main) {
    return enkf_main->res_config;
}

subst_config_type *enkf_main_get_subst_config(const enkf_main_type *enkf_main) {
    return res_config_get_subst_config(enkf_main->res_config);
}

subst_list_type *enkf_main_get_data_kw(const enkf_main_type *enkf_main) {
    return subst_config_get_subst_list(enkf_main_get_subst_config(enkf_main));
}

model_config_type *enkf_main_get_model_config(const enkf_main_type *enkf_main) {
    return res_config_get_model_config(enkf_main->res_config);
}

const ecl_config_type *
enkf_main_get_ecl_config(const enkf_main_type *enkf_main) {
    return res_config_get_ecl_config(enkf_main->res_config);
}

int enkf_main_get_history_length(const enkf_main_type *enkf_main) {
    return model_config_get_last_history_restart(
        enkf_main_get_model_config(enkf_main));
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

void enkf_main_alloc_obs(enkf_main_type *enkf_main) {
    const ecl_config_type *ecl_config = enkf_main_get_ecl_config(enkf_main);
    model_config_type *model_config = enkf_main_get_model_config(enkf_main);
    enkf_main->obs = enkf_obs_alloc(
        model_config_get_history(model_config),
        model_config_get_external_time_map(model_config),
        ecl_config_get_grid(ecl_config), ecl_config_get_refcase(ecl_config),
        enkf_main_get_ensemble_config(enkf_main));
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
                      enkf_main_get_analysis_config(enkf_main)));
    return true;
}

static void enkf_main_add_internal_subst_kw(enkf_main_type *enkf_main,
                                            const char *key, const char *value,
                                            const char *help_text) {
    subst_config_add_internal_subst_kw(enkf_main_get_subst_config(enkf_main),
                                       key, value, help_text);
}

void enkf_main_free(enkf_main_type *enkf_main) {
    if (enkf_main->rng_manager)
        rng_manager_free(enkf_main->rng_manager);

    if (enkf_main->shared_rng)
        rng_free(enkf_main->shared_rng);

    if (enkf_main->obs)
        enkf_obs_free(enkf_main->obs);

    enkf_main_free_ensemble(enkf_main);
    enkf_main_close_fs(enkf_main);

    delete enkf_main;
}

void enkf_main_exit(enkf_main_type *enkf_main) {
    enkf_main_free(enkf_main);
    exit(0);
}

ert_run_context_type *enkf_main_alloc_ert_run_context_ENSEMBLE_EXPERIMENT(
    const enkf_main_type *enkf_main, enkf_fs_type *fs,
    bool_vector_type *iactive, int iter) {
    const model_config_type *model_config =
        enkf_main_get_model_config(enkf_main);
    return ert_run_context_alloc_ENSEMBLE_EXPERIMENT(
        fs, iactive, model_config_get_runpath_fmt(model_config),
        model_config_get_jobname_fmt(model_config),
        enkf_main_get_data_kw(enkf_main), iter);
}

void enkf_main_add_data_kw(enkf_main_type *enkf_main, const char *key,
                           const char *value) {
    subst_config_add_subst_kw(enkf_main_get_subst_config(enkf_main), key,
                              value);
}

static enkf_main_type *enkf_main_alloc_empty() {
    enkf_main_type *enkf_main = new enkf_main_type;
    UTIL_TYPE_ID_INIT(enkf_main, ENKF_MAIN_ID);
    enkf_main->dbase = NULL, enkf_main->ensemble = NULL;
    enkf_main->rng_manager = NULL;
    enkf_main->shared_rng = NULL;
    enkf_main->ens_size = 0;
    enkf_main->res_config = NULL;
    enkf_main->obs = NULL;

    return enkf_main;
}

runpath_list_type *enkf_main_get_runpath_list(const enkf_main_type *enkf_main) {
    return hook_manager_get_runpath_list(enkf_main_get_hook_manager(enkf_main));
}

rng_config_type *enkf_main_get_rng_config(const enkf_main_type *enkf_main) {
    return res_config_get_rng_config(enkf_main->res_config);
}

rng_type *enkf_main_get_shared_rng(enkf_main_type *enkf_main) {
    return enkf_main->shared_rng;
}

void enkf_main_rng_init(enkf_main_type *enkf_main) {
    enkf_main->rng_manager =
        rng_config_alloc_rng_manager(enkf_main_get_rng_config(enkf_main));
    enkf_main->shared_rng = rng_manager_alloc_rng(enkf_main->rng_manager);
}

static void enkf_main_init_obs(enkf_main_type *enkf_main) {
    enkf_main_alloc_obs(enkf_main);

    const model_config_type *model_config =
        enkf_main_get_model_config(enkf_main);
    const char *obs_config_file =
        model_config_get_obs_config_file(model_config);
    if (obs_config_file)
        enkf_main_load_obs(enkf_main, obs_config_file, true);
}

static void enkf_main_add_ensemble_members(enkf_main_type *enkf_main) {
    const model_config_type *model_config =
        enkf_main_get_model_config(enkf_main);
    int num_realizations = model_config_get_num_realizations(model_config);
    enkf_main_increase_ensemble(enkf_main, num_realizations);
}

/**
   This function boots everything needed for running a EnKF
   application from the provided res_config.

  Observe that the function will start with chdir() to the working directory
  specified by res_config, so that all subsequent file
  references are relative to the location of the configuration
  file. This also applies if the command_line argument given is a
  symlink.

    FORWARD_MODEL
    DATA_FILE
    SCHEDULE_FILE
    ECLBASE

   It is possible to pass NULL as the model_config argument, in that
   case only the site config file will be parsed. The purpose of this
   is mainly to be able to test that the site config file is valid.
*/
enkf_main_type *enkf_main_alloc(const res_config_type *res_config) {
    enkf_main_type *enkf_main = enkf_main_alloc_empty();
    enkf_main->res_config = res_config;

    enkf_main_rng_init(enkf_main);
    enkf_main_user_select_initial_fs(enkf_main);
    enkf_main_init_obs(enkf_main);
    enkf_main_add_ensemble_members(enkf_main);

    return enkf_main;
}

int enkf_main_get_ensemble_size(const enkf_main_type *enkf_main) {
    return enkf_main->ens_size;
}

/** @brief initializes internalization.

   The function iterates over all the observations, and ensure that the
   observed nodes (i.e. the pressure for an RFT) are internalized

   Internalize means loaded from the forward simulation and stored in the
   enkf_fs 'database'.
*/
void enkf_main_init_internalization(enkf_main_type *enkf_main) {
    hash_type *map = enkf_obs_alloc_data_map(enkf_main->obs);
    hash_iter_type *iter = hash_iter_alloc(map);
    const char *obs_key = hash_iter_get_next_key(iter);

    while (obs_key != NULL) {
        obs_vector_type *obs_vector =
            enkf_obs_get_vector(enkf_main->obs, obs_key);
        enkf_config_node_type *data_node =
            obs_vector_get_config_node(obs_vector);
        int active_step = -1;
        do {
            active_step =
                obs_vector_get_next_active_step(obs_vector, active_step);
            if (active_step >= 0)
                enkf_config_node_set_internalize(data_node, active_step);
        } while (active_step >= 0);
        obs_key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
    hash_free(map);
}

void enkf_main_get_observations(const enkf_main_type *enkf_main,
                                const char *user_key, int obs_count,
                                time_t *obs_time, double *y, double *std) {
    ensemble_config_get_observations(enkf_main_get_ensemble_config(enkf_main),
                                     enkf_main->obs, user_key, obs_count,
                                     obs_time, y, std);
}

int enkf_main_get_observation_count(const enkf_main_type *enkf_main,
                                    const char *user_key) {
    return ensemble_config_get_observations(
        enkf_main_get_ensemble_config(enkf_main), enkf_main->obs, user_key, 0,
        NULL, NULL, NULL);
}

void enkf_main_install_SIGNALS(void) { util_install_signals(); }

ert_workflow_list_type *enkf_main_get_workflow_list(enkf_main_type *enkf_main) {
    return res_config_get_workflow_list(enkf_main->res_config);
}

int enkf_main_load_from_forward_model_with_fs(enkf_main_type *enkf_main,
                                              int iter,
                                              bool_vector_type *iactive,
                                              enkf_fs_type *fs) {
    model_config_type *model_config = enkf_main_get_model_config(enkf_main);
    ert_run_context_type *run_context =
        ert_run_context_alloc_ENSEMBLE_EXPERIMENT(
            fs, iactive, model_config_get_runpath_fmt(model_config),
            model_config_get_jobname_fmt(model_config),
            enkf_main_get_data_kw(enkf_main), iter);
    int loaded = enkf_main_load_from_run_context(enkf_main, run_context, fs);
    ert_run_context_free(run_context);
    return loaded;
}

int enkf_main_load_from_run_context(enkf_main_type *enkf_main,
                                    ert_run_context_type *run_context,
                                    enkf_fs_type *fs) {
    auto const ens_size = enkf_main_get_ensemble_size(enkf_main);
    auto const *iactive = ert_run_context_get_iactive(run_context);

    // Loading state from a fwd-model is mainly io-bound so we can
    // allow a lot more than #cores threads to execute in parallel.
    // The number 100 is quite arbitrarily chosen though and should
    // probably come from some resource like a site-config or similar.
    // NOTE that this mechanism only limits the number of *concurrently
    // executing* threads. The number of instantiated and stored futures
    // will be equal to the number of active realizations.
    Semafoor concurrently_executing_threads(100);
    std::vector<std::tuple<int, std::future<fw_load_status>>> futures;

    // If this function is called via pybind11 we need to release
    // the GIL here because this function may spin up several
    // threads which also may need the GIL (e.g. for logging)
    PyThreadState *state = nullptr;
    if (PyGILState_Check() == 1)
        state = PyEval_SaveThread();

    for (int iens = 0; iens < ens_size; ++iens) {
        if (bool_vector_iget(iactive, iens)) {

            futures.push_back(std::make_tuple(
                iens, // for logging later
                std::async(
                    std::launch::async,
                    [=](const int realisation, Semafoor &execution_limiter) {
                        // Acquire permit from semaphore or pause execution
                        // until one becomes available. A successfully acquired
                        // permit is released when exiting scope.
                        std::scoped_lock lock(execution_limiter);

                        auto *state_map = enkf_fs_get_state_map(
                            run_arg_get_sim_fs(ert_run_context_iget_arg(
                                run_context, realisation)));

                        state_map_update_undefined(state_map, realisation,
                                                   STATE_INITIALIZED);
                        try {
                            return enkf_state_load_from_forward_model(
                                enkf_main_iget_state(enkf_main, realisation),
                                ert_run_context_iget_arg(run_context,
                                                         realisation));
                        } catch (const std::invalid_argument) {
                            state_map_iset(state_map, realisation,
                                           STATE_LOAD_FAILURE);
                            return LOAD_FAILURE;
                        }
                    },
                    iens, std::ref(concurrently_executing_threads))));
        }
    }

    int loaded = 0;
    for (auto &[iens, fut] : futures) {
        int result = fut.get();
        if (result == LOAD_SUCCESSFUL) {
            loaded++;
        } else if (result == LOAD_FAILURE) {
            logger->warning("Function {}: Realization {} load failure",
                            __func__, iens);
        } else
            logger->error("Unknown load enum");
    }
    if (state)
        PyEval_RestoreThread(state);

    return loaded;
}

bool enkf_main_export_field(const enkf_main_type *enkf_main, const char *kw,
                            const char *path, bool_vector_type *iactive,
                            field_file_format_type file_type, int report_step) {
    enkf_fs_type *fs = enkf_main_get_fs(enkf_main);
    bool result = enkf_main_export_field_with_fs(enkf_main, kw, path, iactive,
                                                 file_type, report_step, fs);
    return result;
}

bool enkf_main_export_field_with_fs(const enkf_main_type *enkf_main,
                                    const char *kw, const char *path,
                                    bool_vector_type *iactive,
                                    field_file_format_type file_type,
                                    int report_step, enkf_fs_type *fs) {

    const ensemble_config_type *ensemble_config =
        enkf_main_get_ensemble_config(enkf_main);
    if (!ensemble_config_has_key(ensemble_config, kw))
        return false;

    enkf_config_node_type *config_node =
        ensemble_config_get_node(ensemble_config, kw);
    if (enkf_config_node_get_impl_type(config_node) != FIELD)
        return false;

    if (util_int_format_count(path) < 1)
        return false;

    enkf_node_type *node = enkf_node_alloc(config_node);
    model_config_type *mc = enkf_main_get_model_config(enkf_main);
    path_fmt_type *runpath_fmt = model_config_get_runpath_fmt(mc);
    const char *init_file =
        enkf_config_node_get_FIELD_fill_file(config_node, runpath_fmt);
    if (init_file)
        printf("init_file found: \"%s\", exporting initial value for inactive "
               "cells\n",
               init_file);
    else
        printf("no init_file found, exporting 0 or fill value for inactive "
               "cells\n");

    for (int iens = 0; iens < bool_vector_size(iactive); ++iens) {
        if (!bool_vector_iget(iactive, iens))
            continue;

        node_id_type node_id = {.report_step = report_step, .iens = iens};
        if (!enkf_node_try_load(node, fs, node_id))
            continue;

        path_fmt_type *export_path = path_fmt_alloc_path_fmt(path);
        char *filename = path_fmt_alloc_path(export_path, false, iens);
        path_fmt_free(export_path);

        char *path;
        util_alloc_file_components(filename, &path, NULL, NULL);
        if (path) {
            util_make_path(path);
            free(path);
        }

        const field_type *field = (const field_type *)enkf_node_value_ptr(node);
        field_export(field, filename, NULL, file_type,
                     true, //output_transform
                     init_file);

        free(filename);
    }
    enkf_node_free(node);

    return true;
}

queue_config_type *enkf_main_get_queue_config(enkf_main_type *enkf_main) {
    return res_config_get_queue_config(enkf_main_get_res_config(enkf_main));
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
        enkf_main_get_ensemble_config(enkf_main), PARAMETER);
    for (auto &key : keylist)
        /*
          Make sure the funny GEN_KW instance masquerading as
          SCHEDULE_PREDICTION_FILE is not added to the soup.
                */
        if (key != "PRED")
            parameters.push_back(key);
    return parameters;
}

int load_from_forward_model_with_fs_pybind(py::object self, int iter,
                                           py::object iactive, py::object fs) {
    auto enkf_main = ert::from_cwrap<enkf_main_type>(self);
    auto iactive_ = ert::from_cwrap<bool_vector_type>(iactive);
    auto fs_ = ert::from_cwrap<enkf_fs_type>(fs);
    return enkf_main_load_from_forward_model_with_fs(enkf_main, iter, iactive_,
                                                     fs_);
}

/**
  This function writes out all the files needed by an ECLIPSE simulation, this
  includes the restart file, and the various INCLUDE files corresponding to
  parameters estimated by EnKF.

  The writing of restart file is delegated to enkf_state_write_restart_file().

  TODO: enkf_fs_type could be fetched from run_arg
*/
void enkf_state_ecl_write(const ensemble_config_type *ens_config,
                          const model_config_type *model_config,
                          const run_arg_type *run_arg, enkf_fs_type *fs) {
    /*
     This iteration manipulates the hash (thorugh the enkf_state_del_node() call)

     -----------------------------------------------------------------------------------------
     T H I S  W I L L  D E A D L O C K  I F  T H E   H A S H _ I T E R  A P I   I S   U S E D.
     -----------------------------------------------------------------------------------------
  */
    int iens = run_arg_get_iens(run_arg);
    const char *base_name = model_config_get_gen_kw_export_name(model_config);
    value_export_type *export_value =
        value_export_alloc(run_arg_get_runpath(run_arg), base_name);

    std::vector<std::string> key_list = ensemble_config_keylist_from_var_type(
        ens_config, PARAMETER + EXT_PARAMETER);
    for (auto &key : key_list) {
        enkf_config_node_type *config_node =
            ensemble_config_get_node(ens_config, key.c_str());
        enkf_node_type *enkf_node = enkf_node_alloc(config_node);
        bool forward_init = enkf_node_use_forward_init(enkf_node);
        node_id_type node_id = {.report_step = run_arg_get_step1(run_arg),
                                .iens = iens};

        if ((run_arg_get_step1(run_arg) == 0) && (forward_init)) {

            if (enkf_node_has_data(enkf_node, fs, node_id))
                enkf_node_load(enkf_node, fs, node_id);
            else
                continue;
        } else
            enkf_node_load(enkf_node, fs, node_id);

        enkf_node_ecl_write(enkf_node, run_arg_get_runpath(run_arg),
                            export_value, run_arg_get_step1(run_arg));
        enkf_node_free(enkf_node);
    }
    value_export(export_value);

    value_export_free(export_value);
}

void enkf_state_init_eclipse(const res_config_type *res_config,
                             const run_arg_type *run_arg) {

    ensemble_config_type *ens_config =
        res_config_get_ensemble_config(res_config);
    const ecl_config_type *ecl_config = res_config_get_ecl_config(res_config);
    model_config_type *model_config = res_config_get_model_config(res_config);

    util_make_path(run_arg_get_runpath(run_arg));

    ert_templates_instansiate(res_config_get_templates(res_config),
                              run_arg_get_runpath(run_arg),
                              run_arg_get_subst_list(run_arg));

    enkf_state_ecl_write(ens_config, model_config, run_arg,
                         run_arg_get_sim_fs(run_arg));

    /* Writing the ECLIPSE data file. */
    if (ecl_config_have_eclbase(ecl_config) &&
        ecl_config_get_data_file(ecl_config)) {
        char *data_file = ecl_util_alloc_filename(run_arg_get_runpath(run_arg),
                                                  run_arg_get_job_name(run_arg),
                                                  ECL_DATA_FILE, true, -1);

        subst_list_update_string(run_arg_get_subst_list(run_arg), &data_file);
        subst_list_filter_file(run_arg_get_subst_list(run_arg),
                               ecl_config_get_data_file(ecl_config), data_file);

        free(data_file);
    }

    mode_t umask =
        site_config_get_umask(res_config_get_site_config(res_config));

    /* This is where the job script is created */
    const env_varlist_type *varlist =
        site_config_get_env_varlist(res_config_get_site_config(res_config));
    forward_model_formatted_fprintf(
        model_config_get_forward_model(model_config),
        run_arg_get_run_id(run_arg), run_arg_get_runpath(run_arg),
        model_config_get_data_root(model_config),
        run_arg_get_subst_list(run_arg), umask, varlist);
}

void enkf_main_write_run_path(enkf_main_type *enkf_main,
                              const ert_run_context_type *run_context) {
    runpath_list_type *runpath_list = enkf_main_get_runpath_list(enkf_main);
    runpath_list_clear(runpath_list);
    for (int iens = 0; iens < ert_run_context_get_size(run_context); iens++) {
        if (ert_run_context_iactive(run_context, iens)) {
            run_arg_type *run_arg = ert_run_context_iget_arg(run_context, iens);
            runpath_list_add(runpath_list, run_arg_get_iens(run_arg),
                             run_arg_get_iter(run_arg),
                             run_arg_get_runpath(run_arg),
                             run_arg_get_job_name(run_arg));
            enkf_state_init_eclipse(enkf_main->res_config, run_arg);
        }
    }
    runpath_list_fprintf(runpath_list);
}

RES_LIB_SUBMODULE("enkf_main", m) {
    using namespace py::literals;
    m.def("get_observation_keys", get_observation_keys);
    m.def("get_parameter_keys", get_parameter_keys);
    m.def(
        "init_internalization",
        [](py::object self) {
            auto enkf_main = ert::from_cwrap<enkf_main_type>(self);
            return enkf_main_init_internalization(enkf_main);
        },
        py::arg("self"));
    m.def(
        "write_run_path",
        [](py::object self, py::object run_context_py) {
            auto enkf_main = ert::from_cwrap<enkf_main_type>(self);
            auto run_context =
                ert::from_cwrap<ert_run_context_type>(run_context_py);
            return enkf_main_write_run_path(enkf_main, run_context);
        },
        py::arg("self"), py::arg("run_context"));
    m.def("load_from_forward_model", load_from_forward_model_with_fs_pybind,
          py::arg("self"), py::arg("iter"), py::arg("iactive"), py::arg("fs"));
}

#include "enkf_main_ensemble.cpp"
#include "enkf_main_manage_fs.cpp"
