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
#include <ert/util/type_vector_functions.h>
#include <ert/util/vector.hpp>

#include <ert/logging.hpp>
#include <ert/res_util/subst_list.hpp>

#include <ert/sched/history.hpp>

#include <ert/analysis/analysis_module.hpp>
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

struct enkf_main_struct {
    UTIL_TYPE_ID_DECLARATION;
    /** The internalized information. */

    const res_config_type *res_config;

    enkf_obs_type *obs;

    /** The size of the ensemble */
    int ens_size;
};

UTIL_SAFE_CAST_FUNCTION(enkf_main, ENKF_MAIN_ID)
UTIL_IS_INSTANCE_FUNCTION(enkf_main, ENKF_MAIN_ID)

const res_config_type *
enkf_main_get_res_config(const enkf_main_type *enkf_main) {
    return enkf_main->res_config;
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

void enkf_main_free(enkf_main_type *enkf_main) {

    if (enkf_main->obs)
        enkf_obs_free(enkf_main->obs);

    delete enkf_main;
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

int enkf_main_load_from_run_context(const int ens_size,
                                    ensemble_config_type *ensemble_config,
                                    model_config_type *model_config,
                                    ecl_config_type *ecl_config,
                                    std::vector<bool> active_mask,
                                    enkf_fs_type *sim_fs,
                                    std::vector<run_arg_type *> run_args) {
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

                        state_map.update_matching(realisation, STATE_UNDEFINED,
                                                  STATE_INITIALIZED);
                        auto status = enkf_state_load_from_forward_model(
                            ensemble_config, model_config, ecl_config,
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
    return ensemble_config_keylist_from_var_type(
        res_config_get_ensemble_config(enkf_main->res_config), PARAMETER);
}

namespace enkf_main {
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

enkf_main_type *enkf_main_alloc(const res_config_type *res_config,
                                bool read_only) {
    const ecl_config_type *ecl_config = res_config_get_ecl_config(res_config);
    const model_config_type *model_config =
        res_config_get_model_config(res_config);

    enkf_main_type *enkf_main = new enkf_main_type;
    UTIL_TYPE_ID_INIT(enkf_main, ENKF_MAIN_ID);

    enkf_main->res_config = res_config;

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

    return enkf_main;
}

ERT_CLIB_SUBMODULE("enkf_main", m) {
    using namespace py::literals;
    m.def(
        "init_current_case_from_existing_custom",
        [](py::object ensemble_config_py, py::object source_case_py,
           py::object current_case_py, int source_report_step,
           std::vector<std::string> &node_list, std::vector<bool> &iactive) {
            auto source_case_fs = ert::from_cwrap<enkf_fs_type>(source_case_py);
            auto current_fs = ert::from_cwrap<enkf_fs_type>(current_case_py);
            auto ensemble_config =
                ert::from_cwrap<ensemble_config_type>(ensemble_config_py);
            enkf_main_copy_ensemble(ensemble_config, source_case_fs,
                                    source_report_step, current_fs, iactive,
                                    node_list);
            enkf_fs_fsync(current_fs);
        },
        py::arg("self"), py::arg("source_case"), py::arg("current_case"),
        py::arg("source_report_step"), py::arg("node_list"),
        py::arg("iactive"));
    m.def("get_observation_keys", get_observation_keys);
    m.def("get_parameter_keys", get_parameter_keys);
    m.def("load_from_run_context",
          [](int ens_size, py::object ensemble_config, py::object model_config,
             py::object ecl_config, std::vector<py::object> run_args_,
             std::vector<bool> active_mask, py::object sim_fs_) {
              auto sim_fs = ert::from_cwrap<enkf_fs_type>(sim_fs_);
              std::vector<run_arg_type *> run_args;
              for (auto &run_arg : run_args_) {
                  run_args.push_back(ert::from_cwrap<run_arg_type>(run_arg));
              }
              return enkf_main_load_from_run_context(
                  ens_size,
                  ert::from_cwrap<ensemble_config_type>(ensemble_config),
                  ert::from_cwrap<model_config_type>(model_config),
                  ert::from_cwrap<ecl_config_type>(ecl_config), active_mask,
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
}
