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

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <thread>

#include <unordered_map>
#include <vector>

#define HAVE_THREAD_POOL 1
#include <ert/util/rng.h>
#include <ert/util/vector.hpp>
#include <ert/util/int_vector.h>
#include <ert/util/bool_vector.h>
#include <ert/util/hash.h>
#include <ert/res_util/path_fmt.hpp>
#include <ert/res_util/arg_pack.hpp>
#include <ert/util/type_vector_functions.h>

#include <ert/res_util/thread_pool.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/res_util/res_log.hpp>
#include <ert/res_util/matrix.hpp>

#include <ert/job_queue/job_queue.hpp>

#include <ert/sched/history.hpp>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/enkf_linalg.hpp>
#include <ert/analysis/update.hpp>

#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/enkf_main.hpp>
#include <ert/enkf/enkf_analysis.hpp>
#include <ert/enkf/field.hpp>
#include <ert/enkf/callback_arg.hpp>

/*
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

#define ENKF_MAIN_ID 8301

struct enkf_main_struct {
    UTIL_TYPE_ID_DECLARATION;
    enkf_fs_type *dbase; /* The internalized information. */

    const res_config_type *res_config;
    local_config_type
        *local_config; /* Holding all the information about local analysis. */
    rng_manager_type *rng_manager;
    rng_type *shared_rng;
    ranking_table_type *ranking_table;

    enkf_obs_type *obs;

    enkf_state_type **ensemble; /* The ensemble ... */
    int ens_size;               /* The size of the ensemble */
    bool verbose;
};

void enkf_main_init_internalization(enkf_main_type *);
void enkf_main_update_local_updates(enkf_main_type *enkf_main);
static void enkf_main_close_fs(enkf_main_type *enkf_main);
static void enkf_main_init_fs(enkf_main_type *enkf_main);
static void enkf_main_user_select_initial_fs(enkf_main_type *enkf_main);
static void enkf_main_free_ensemble(enkf_main_type *enkf_main);

UTIL_SAFE_CAST_FUNCTION(enkf_main, ENKF_MAIN_ID)
UTIL_IS_INSTANCE_FUNCTION(enkf_main, ENKF_MAIN_ID)

const analysis_config_type *
enkf_main_get_analysis_config(const enkf_main_type *enkf_main) {
    return res_config_get_analysis_config(enkf_main->res_config);
}

const char *enkf_main_get_user_config_file(const enkf_main_type *enkf_main) {
    return res_config_get_user_config_file(enkf_main->res_config);
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

const log_config_type *
enkf_main_get_log_config(const enkf_main_type *enkf_main) {
    return res_config_get_log_config(enkf_main->res_config);
}

subst_config_type *enkf_main_get_subst_config(const enkf_main_type *enkf_main) {
    return res_config_get_subst_config(enkf_main->res_config);
}

subst_list_type *enkf_main_get_data_kw(const enkf_main_type *enkf_main) {
    return subst_config_get_subst_list(enkf_main_get_subst_config(enkf_main));
}

local_config_type *enkf_main_get_local_config(const enkf_main_type *enkf_main) {
    return enkf_main->local_config;
}

model_config_type *enkf_main_get_model_config(const enkf_main_type *enkf_main) {
    return res_config_get_model_config(enkf_main->res_config);
}

ranking_table_type *
enkf_main_get_ranking_table(const enkf_main_type *enkf_main) {
    return enkf_main->ranking_table;
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
    enkf_main_update_local_updates(enkf_main);
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

    ranking_table_free(enkf_main->ranking_table);
    enkf_main_free_ensemble(enkf_main);
    enkf_main_close_fs(enkf_main);
    res_log_close();

    local_config_free(enkf_main->local_config);

    free(enkf_main);
}

void enkf_main_exit(enkf_main_type *enkf_main) {
    enkf_main_free(enkf_main);
    exit(0);
}

// Opens and returns a log file.  A subroutine of enkf_main_smoother_update.
static FILE *enkf_main_log_step_list(const char *log_path,
                                     const std::vector<int> &step_list) {
    char *log_file;
    if (step_list.size() == 1)
        log_file = util_alloc_sprintf("%s%c%04d", log_path, UTIL_PATH_SEP_CHAR,
                                      step_list.front());
    else
        log_file =
            util_alloc_sprintf("%s%c%04d-%04d", log_path, UTIL_PATH_SEP_CHAR,
                               step_list.front(), step_list.back());
    FILE *log_stream = util_fopen(log_file, "w");

    free(log_file);
    return log_stream;
}

bool enkf_main_smoother_update(enkf_main_type *enkf_main,
                               enkf_fs_type *source_fs,
                               enkf_fs_type *target_fs) {
    time_map_type *time_map = enkf_fs_get_time_map(source_fs);

    int last_step = time_map_get_last_step(time_map);
    if (last_step < 0)
        last_step = model_config_get_last_history_restart(
            enkf_main_get_model_config(enkf_main));

    std::vector<int> step_list;
    for (int i = 0; i <= last_step; i++)
        step_list.push_back(i);

    local_config_type *local_config = enkf_main->local_config;
    const local_updatestep_type *updatestep =
        local_config_get_updatestep(local_config);
    const analysis_config_type *analysis_config =
        enkf_main_get_analysis_config(enkf_main);

    const int total_ens_size = enkf_main->ens_size;
    enkf_obs_type *obs = enkf_main->obs;
    bool verbose = enkf_main->verbose;
    rng_type *shared_rng = enkf_main->shared_rng;
    ensemble_config_type *ensemble_config =
        enkf_main_get_ensemble_config(enkf_main);
    FILE *log_stream = enkf_main_log_step_list(
        analysis_config_get_log_path(analysis_config), step_list);

    bool ok = analysis::smoother_update(
        step_list, updatestep, total_ens_size, obs, shared_rng, analysis_config,
        ensemble_config, source_fs, target_fs, log_stream, verbose);

    fclose(log_stream);
    return ok;
}

static void enkf_main_write_run_path(enkf_main_type *enkf_main,
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

void enkf_main_create_run_path(enkf_main_type *enkf_main,
                               const ert_run_context_type *run_context) {
    enkf_main_init_run(enkf_main, run_context);
    enkf_main_write_run_path(enkf_main, run_context);
}

/*
   This function will initialize the necessary enkf_main structures
   before a run. Currently this means:

     1. Set the enkf_sched instance - either by loading from file or
        by using the default.

     2. Set up the configuration of what should be internalized.

*/

void enkf_main_init_run(enkf_main_type *enkf_main,
                        const ert_run_context_type *run_context) {
    enkf_main_init_internalization(enkf_main);
    {
        stringlist_type *param_list =
            ensemble_config_alloc_keylist_from_var_type(
                enkf_main_get_ensemble_config(enkf_main), PARAMETER);
        enkf_main_initialize_from_scratch(enkf_main, param_list, run_context);
        stringlist_free(param_list);
    }
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

/*
   This function creates a local_config file corresponding to the
   default 'ALL_ACTIVE' configuration.
*/

void enkf_main_create_all_active_config(const enkf_main_type *enkf_main) {

    bool single_node_update = analysis_config_get_single_node_update(
        enkf_main_get_analysis_config(enkf_main));
    local_config_type *local_config = enkf_main->local_config;
    local_config_clear(local_config);
    {
        local_updatestep_type *default_step =
            local_config_get_updatestep(local_config);
        local_obsdata_type *obsdata =
            local_config_alloc_obsdata(local_config, "ALL_OBS");
        local_dataset_type *all_active_dataset =
            local_config_alloc_dataset(local_config, "ALL_DATA");
        local_ministep_type *ministep =
            local_config_alloc_ministep(local_config, "ALL_ACTIVE", NULL);
        if (!ministep)
            throw std::logic_error(
                "Failed to create initial ALL_ACTIVE ministep");

        local_updatestep_add_ministep(default_step, ministep);

        /* Adding all observation keys */
        {
            hash_iter_type *obs_iter = enkf_obs_alloc_iter(enkf_main->obs);
            while (!hash_iter_is_complete(obs_iter)) {
                const char *obs_key = hash_iter_get_next_key(obs_iter);
                local_obsdata_node_type *obsdata_node =
                    local_obsdata_node_alloc(obs_key, true);
                local_obsdata_add_node(obsdata, obsdata_node);
            }
            local_ministep_add_obsdata(ministep, obsdata);
            hash_iter_free(obs_iter);
        }

        /* Adding all node which can be updated. */
        {
            stringlist_type *keylist =
                ensemble_config_alloc_keylist_from_var_type(
                    enkf_main_get_ensemble_config(enkf_main), PARAMETER);
            int i;
            for (i = 0; i < stringlist_get_size(keylist); i++) {
                const char *key = stringlist_iget(keylist, i);
                bool add_node = true;

                /*
          Make sure the funny GEN_KW instance masquerading as
          SCHEDULE_PREDICTION_FILE is not added to the soup.
        */
                if (util_string_equal(key, "PRED"))
                    add_node = false;

                if (add_node) {
                    if (single_node_update) {
                        local_dataset_type *this_dataset =
                            local_config_alloc_dataset(local_config, key);
                        local_dataset_add_node(this_dataset, key);
                        local_ministep_add_dataset(ministep, this_dataset);
                    }
                    local_dataset_add_node(all_active_dataset, key);
                }
            }
            stringlist_free(keylist);
        }
        if (!single_node_update)
            local_ministep_add_dataset(ministep, all_active_dataset);
    }
}

void enkf_main_set_verbose(enkf_main_type *enkf_main, bool verbose) {
    enkf_main->verbose = verbose;
}

/*
   There is NO tagging anymore - if the user wants tags - the user
   supplies the key __WITH__ tags.
*/
void enkf_main_add_data_kw(enkf_main_type *enkf_main, const char *key,
                           const char *value) {
    subst_config_add_subst_kw(enkf_main_get_subst_config(enkf_main), key,
                              value);
}

void enkf_main_clear_data_kw(enkf_main_type *enkf_main) {
    subst_config_clear(enkf_main_get_subst_config(enkf_main));
}

static enkf_main_type *enkf_main_alloc_empty() {
    enkf_main_type *enkf_main =
        (enkf_main_type *)util_malloc(sizeof *enkf_main);
    UTIL_TYPE_ID_INIT(enkf_main, ENKF_MAIN_ID);
    enkf_main->ensemble = NULL;
    enkf_main->local_config = NULL;
    enkf_main->rng_manager = NULL;
    enkf_main->shared_rng = NULL;
    enkf_main->ens_size = 0;
    enkf_main->res_config = NULL;
    enkf_main->ranking_table = ranking_table_alloc(0);
    enkf_main->obs = NULL;
    enkf_main->local_config = local_config_alloc();

    enkf_main_set_verbose(enkf_main, false);
    enkf_main_init_fs(enkf_main);

    return enkf_main;
}

runpath_list_type *enkf_main_get_runpath_list(const enkf_main_type *enkf_main) {
    return hook_manager_get_runpath_list(enkf_main_get_hook_manager(enkf_main));
}

runpath_list_type *
enkf_main_alloc_runpath_list(const enkf_main_type *enkf_main) {
    return runpath_list_alloc(hook_manager_get_runpath_list_file(
        enkf_main_get_hook_manager(enkf_main)));
}

void enkf_main_add_node(enkf_main_type *enkf_main,
                        enkf_config_node_type *enkf_config_node) {
    for (int iens = 0; iens < enkf_main_get_ensemble_size(enkf_main); iens++) {

        enkf_state_add_node(enkf_main->ensemble[iens],
                            enkf_config_node_get_key(enkf_config_node),
                            enkf_config_node);
    }
}

const char *
enkf_main_get_schedule_prediction_file(const enkf_main_type *enkf_main) {
    return ecl_config_get_schedule_prediction_file(
        enkf_main_get_ecl_config(enkf_main));
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

void enkf_main_update_local_updates(enkf_main_type *enkf_main) {
    const enkf_obs_type *enkf_obs = enkf_main_get_obs(enkf_main);
    if (enkf_obs_have_obs(enkf_obs)) {
        /* First create the default ALL_ACTIVE configuration. */
        enkf_main_create_all_active_config(enkf_main);
    }
}

static void enkf_main_init_log(const enkf_main_type *enkf_main) {
    const log_config_type *log_config = enkf_main_get_log_config(enkf_main);
    res_log_init_log(log_config_get_log_level(log_config),
                     log_config_get_log_file(log_config), enkf_main->verbose);
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
    enkf_main_resize_ensemble(enkf_main, num_realizations);
}

/*
   This function boots everything needed for running a EnKF
   application from the provided res_config.

  Observe that the function will start with chdir() to the working directory
  specified by res_config, so that all subsequent file
  references are relative to the location of the configuration
  file. This also applies if the command_line argument given is a
  symlink.


  If the parameter @strict is set to false a configuration with some
  missing parameters will validate; this is to support bootstrapping
  from a minimal configuration created by the GUI. The parameters
  which become optional in a non-strict mode are:

    FORWARD_MODEL
    DATA_FILE
    SCHEDULE_FILE
    ECLBASE

*/

/*
   It is possible to pass NULL as the model_config argument, in that
   case only the site config file will be parsed. The purpose of this
   is mainly to be able to test that the site config file is valid.
*/

enkf_main_type *enkf_main_alloc(const res_config_type *res_config, bool strict,
                                bool verbose) {
    enkf_main_type *enkf_main = enkf_main_alloc_empty();
    enkf_main->res_config = res_config;

    enkf_main_set_verbose(enkf_main, verbose);
    enkf_main_init_log(enkf_main);
    enkf_main_rng_init(enkf_main);
    enkf_main_user_select_initial_fs(enkf_main);
    enkf_main_init_obs(enkf_main);
    enkf_main_add_ensemble_members(enkf_main);

    return enkf_main;
}

int enkf_main_get_ensemble_size(const enkf_main_type *enkf_main) {
    return enkf_main->ens_size;
}

/*
   In this function we initialize the variables which control
   which nodes are internalized (i.e. loaded from the forward
   simulation and stored in the enkf_fs 'database'). The system is
   based on two-levels:

   * Should we store the state? This is goverened by the variable
     model_config->internalize_state. If this is true we will
     internalize all nodes which have enkf_var_type = {dynamic_state ,
     static_state}. In the same way the variable
     model_config->internalize_results governs whether the dynamic
     results (i.e. summary variables in ECLIPSE speak) should be
     internalized.

   * In addition we have fine-grained control in the enkf_config_node
     objects where we can explicitly say that, altough we do not want
     to internalize the full state, we want to internalize e.g. the
     pressure field.

   * All decisions on internalization are based on a per report step
     basis.

   The user-space API for manipulating this is (extremely)
   limited. What is implemented here is the following:

     1. We internalize the initial dynamic state.

     2. For all the end-points in the current enkf_sched instance we
        internalize the state.

     3. store_results is set to true for all report steps irrespective
        of run_mode.

     4. We iterate over all the observations, and ensure that the
        observed nodes (i.e. the pressure for an RFT) are internalized
        (irrespective of whether they are of type dynamic_state or
        dynamic_result).

   Observe that this cascade can result in some nodes, i.e. a rate we
   are observing, to be marked for internalization several times -
   that is no problem.

   -----

   For performance reason model_config contains the bool vector
   __load_eclipse_restart; if it is true the ECLIPSE restart state is
   loaded from disk, otherwise no loading is performed. This implies
   that if we do not want to internalize the full state but for
   instance the pressure (i.e. for an RFT) we must set the
   __load_state variable for the actual report step to true.
*/

void enkf_main_init_internalization(enkf_main_type *enkf_main) {
    /* Clearing old internalize flags. */
    model_config_init_internalization(enkf_main_get_model_config(enkf_main));

    /* Internalizing the initial state. */
    model_config_set_internalize_state(enkf_main_get_model_config(enkf_main),
                                       0);

    /* Make sure we internalize at all observation times.*/
    {
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

ert_templates_type *enkf_main_get_templates(enkf_main_type *enkf_main) {
    return res_config_get_templates(enkf_main->res_config);
}

ert_workflow_list_type *enkf_main_get_workflow_list(enkf_main_type *enkf_main) {
    return res_config_get_workflow_list(enkf_main->res_config);
}

int enkf_main_load_from_forward_model_from_gui(enkf_main_type *enkf_main,
                                               int iter,
                                               bool_vector_type *iactive,
                                               enkf_fs_type *fs) {
    const int ens_size = enkf_main_get_ensemble_size(enkf_main);
    stringlist_type **realizations_msg_list = (stringlist_type **)util_calloc(
        ens_size, sizeof *realizations_msg_list); // CXX_CAST_ERROR
    for (int iens = 0; iens < ens_size; ++iens)
        realizations_msg_list[iens] = stringlist_alloc_new();

    int loaded = enkf_main_load_from_forward_model_with_fs(
        enkf_main, iter, iactive, realizations_msg_list, fs);

    for (int iens = 0; iens < ens_size; ++iens)
        stringlist_free(realizations_msg_list[iens]);

    free(realizations_msg_list);
    return loaded;
}

int enkf_main_load_from_forward_model(enkf_main_type *enkf_main, int iter,
                                      bool_vector_type *iactive,
                                      stringlist_type **realizations_msg_list) {
    enkf_fs_type *fs = enkf_main_get_fs(enkf_main);
    return enkf_main_load_from_forward_model_with_fs(enkf_main, iter, iactive,
                                                     realizations_msg_list, fs);
}

int enkf_main_load_from_forward_model_with_fs(
    enkf_main_type *enkf_main, int iter, bool_vector_type *iactive,
    stringlist_type **realizations_msg_list, enkf_fs_type *fs) {
    model_config_type *model_config = enkf_main_get_model_config(enkf_main);
    ert_run_context_type *run_context =
        ert_run_context_alloc_ENSEMBLE_EXPERIMENT(
            fs, iactive, model_config_get_runpath_fmt(model_config),
            model_config_get_jobname_fmt(model_config),
            enkf_main_get_data_kw(enkf_main), iter);
    int loaded = enkf_main_load_from_run_context(enkf_main, run_context,
                                                 realizations_msg_list, fs);
    ert_run_context_free(run_context);
    return loaded;
}

int enkf_main_load_from_run_context_from_gui(enkf_main_type *enkf_main,
                                             ert_run_context_type *run_context,
                                             enkf_fs_type *fs) {
    auto const ens_size = enkf_main_get_ensemble_size(enkf_main);
    stringlist_type **realizations_msg_list = (stringlist_type **)util_calloc(
        ens_size, sizeof *realizations_msg_list); // CXX_CAST_ERROR
    for (int iens = 0; iens < ens_size; ++iens)
        realizations_msg_list[iens] = stringlist_alloc_new();

    int loaded = enkf_main_load_from_run_context(enkf_main, run_context,
                                                 realizations_msg_list, fs);

    for (int iens = 0; iens < ens_size; ++iens)
        stringlist_free(realizations_msg_list[iens]);
    free(realizations_msg_list);
    return loaded;
}

int enkf_main_load_from_run_context(enkf_main_type *enkf_main,
                                    ert_run_context_type *run_context,
                                    stringlist_type **realizations_msg_list,
                                    enkf_fs_type *fs) {
    auto const ens_size = enkf_main_get_ensemble_size(enkf_main);
    auto const *iactive = ert_run_context_get_iactive(run_context);

    int result[ens_size];
    arg_pack_type **arg_list = (arg_pack_type **)util_calloc(
        ens_size, sizeof *arg_list); // CXX_CAST_ERROR
    thread_pool_type *tp =
        thread_pool_alloc(std::thread::hardware_concurrency(), true);

    for (int iens = 0; iens < ens_size; ++iens) {
        result[iens] = 0;
        arg_pack_type *arg_pack = arg_pack_alloc();
        arg_list[iens] = arg_pack;

        if (bool_vector_iget(iactive, iens)) {
            enkf_state_type *enkf_state = enkf_main_iget_state(enkf_main, iens);
            arg_pack_append_ptr(arg_pack, enkf_state); /* 0: enkf_state*/
            arg_pack_append_ptr(
                arg_pack,
                ert_run_context_iget_arg(run_context, iens)); /* 1: run_arg */
            arg_pack_append_ptr(
                arg_pack,
                realizations_msg_list
                    [iens]); /* 2: List of interactive mode messages. */
            arg_pack_append_bool(arg_pack, true);         /* 3: Manual load */
            arg_pack_append_ptr(arg_pack, &result[iens]); /* 4: Result */
            thread_pool_add_job(tp, enkf_state_load_from_forward_model_mt,
                                arg_pack);
        }
    }

    thread_pool_join(tp);
    thread_pool_free(tp);

    int loaded = 0;
    for (int iens = 0; iens < ens_size; ++iens) {
        if (bool_vector_iget(iactive, iens)) {
            if (result[iens] & LOAD_FAILURE)
                fprintf(
                    stderr,
                    "** Warning: Function %s: Realization %d load failure\n",
                    __func__, iens);
            else if (result[iens] & REPORT_STEP_INCOMPATIBLE)
                fprintf(stderr,
                        "** Warning: Function %s: Realization %d report step "
                        "incompatible\n",
                        __func__, iens);
            else
                loaded++;
        }
        arg_pack_free(arg_list[iens]);
    }
    free(arg_list);
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

void enkf_main_rank_on_observations(enkf_main_type *enkf_main,
                                    const char *ranking_key,
                                    const stringlist_type *obs_ranking_keys,
                                    const int_vector_type *steps) {

    enkf_fs_type *fs = enkf_main_get_fs(enkf_main);
    const enkf_obs_type *enkf_obs = enkf_main_get_obs(enkf_main);
    const ensemble_config_type *ensemble_config =
        enkf_main_get_ensemble_config(enkf_main);
    const int history_length = enkf_main_get_history_length(enkf_main);
    const int ens_size = enkf_main_get_ensemble_size(enkf_main);

    misfit_ensemble_type *misfit_ensemble = enkf_fs_get_misfit_ensemble(fs);
    misfit_ensemble_initialize(misfit_ensemble, ensemble_config, enkf_obs, fs,
                               ens_size, history_length, false);

    ranking_table_type *ranking_table = enkf_main_get_ranking_table(enkf_main);

    ranking_table_add_misfit_ranking(ranking_table, misfit_ensemble,
                                     obs_ranking_keys, steps, ranking_key);
    ranking_table_display_ranking(ranking_table, ranking_key);
}

void enkf_main_rank_on_data(enkf_main_type *enkf_main, const char *ranking_key,
                            const char *data_key, bool sort_increasing,
                            int step) {

    ranking_table_type *ranking_table = enkf_main_get_ranking_table(enkf_main);
    const ensemble_config_type *ensemble_config =
        enkf_main_get_ensemble_config(enkf_main);
    enkf_fs_type *fs = enkf_main_get_fs(enkf_main);
    char *key_index;

    const enkf_config_node_type *config_node =
        ensemble_config_user_get_node(ensemble_config, data_key, &key_index);
    if (config_node) {
        ranking_table_add_data_ranking(ranking_table, sort_increasing,
                                       ranking_key, data_key, key_index, fs,
                                       config_node, step);
        ranking_table_display_ranking(ranking_table, ranking_key);
    } else {
        fprintf(stderr, "** No data found for key %s\n", data_key);
    }
}

void enkf_main_export_ranking(enkf_main_type *enkf_main,
                              const char *ranking_key,
                              const char *ranking_file) {
    ranking_table_type *ranking_table = enkf_main_get_ranking_table(enkf_main);
    ranking_table_fwrite_ranking(ranking_table, ranking_key, ranking_file);
}

queue_config_type *enkf_main_get_queue_config(enkf_main_type *enkf_main) {
    return res_config_get_queue_config(enkf_main_get_res_config(enkf_main));
}

rng_manager_type *enkf_main_get_rng_manager(const enkf_main_type *enkf_main) {
    return enkf_main->rng_manager;
}

#include "enkf_main_ensemble.cpp"
#include "enkf_main_manage_fs.cpp"
