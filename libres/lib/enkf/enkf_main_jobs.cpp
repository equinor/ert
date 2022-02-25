/*
   Copyright (C) 2012  Equinor ASA, Norway.

   The file 'enkf_main_jobs.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <cassert>

#include <ert/util/stringlist.h>
#include <ert/util/string_util.h>

#include <ert/logging.hpp>
#include <ert/enkf/enkf_main.hpp>
#include <ert/util/type_vector_functions.hpp>

static auto logger = ert::get_logger("enkf");

static bool_vector_type *
alloc_iactive_vector_from_range(const stringlist_type *range, int startindex,
                                int endindex, int ens_size) {
    bool_vector_type *iactive;
    int range_list_size = stringlist_get_size(range);
    if ((range_list_size > startindex) && (range_list_size >= endindex)) {
        char *arg_string =
            stringlist_alloc_joined_substring(range, startindex, endindex, "");
        iactive = bool_vector_alloc(ens_size, false);
        string_util_update_active_mask(arg_string, iactive);
        free(arg_string);
    } else {
        iactive = bool_vector_alloc(ens_size, true);
    }
    return iactive;
}

// Internal workflow job
extern "C" C_USED void *enkf_main_exit_JOB(void *self,
                                           const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    enkf_main_exit(enkf_main);
    return NULL;
}

/*
   Will create the new case if it does not exist.
*/
// Internal workflow job
extern "C" C_USED void *enkf_main_select_case_JOB(void *self,
                                                  const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    const char *new_case = stringlist_iget(args, 0);
    enkf_main_select_fs(enkf_main, new_case);
    return NULL;
}

// Internal workflow job
extern "C" C_USED void *enkf_main_create_case_JOB(void *self,
                                                  const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    const char *new_case = stringlist_iget(args, 0);
    enkf_fs_type *fs = enkf_main_mount_alt_fs(enkf_main, new_case, true);
    enkf_fs_decref(fs);
    return NULL;
}

// Internal workflow job
extern "C" C_USED void *
enkf_main_init_case_from_existing_JOB(void *self, const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);

    const char *source_case = stringlist_iget(args, 0);
    enkf_fs_type *source_fs =
        enkf_main_mount_alt_fs(enkf_main, source_case, true);
    {
        enkf_fs_type *target_fs;

        if (stringlist_get_size(args) > 1) {
            const char *current_case = enkf_main_get_current_fs(enkf_main);
            const char *target_case = stringlist_iget(args, 1);
            if (0 != strcmp(current_case, target_case)) {
                target_fs =
                    enkf_main_mount_alt_fs(enkf_main, target_case, true);
            } else
                target_fs = enkf_fs_get_ref(enkf_main_job_get_fs(
                    enkf_main)); // Using get_ref so that we can unconditionally call decref() further down.
        } else
            target_fs = enkf_fs_get_ref(enkf_main_job_get_fs(
                enkf_main)); // Using get_ref so that we can unconditionally call decref() further down.

        enkf_main_init_case_from_existing(
            enkf_main, source_fs, 0, target_fs); // Removed ANALYZED argument
        enkf_fs_decref(target_fs);
    }
    enkf_fs_decref(source_fs);

    return NULL;
}

static void *enkf_main_load_results_JOB__(enkf_main_type *enkf_main, int iter,
                                          const stringlist_type *args) {
    bool_vector_type *iactive =
        alloc_iactive_vector_from_range(args, 0, stringlist_get_size(args),
                                        enkf_main_get_ensemble_size(enkf_main));
    int ens_size = enkf_main_get_ensemble_size(enkf_main);
    stringlist_type **realizations_msg_list = (stringlist_type **)util_calloc(
        ens_size, sizeof *realizations_msg_list);
    for (int iens = 0; iens < ens_size; ++iens)
        realizations_msg_list[iens] = stringlist_alloc_new();

    enkf_main_load_from_forward_model(enkf_main, iter, iactive,
                                      realizations_msg_list);

    for (int iens = 0; iens < ens_size; ++iens) {
        stringlist_type *msg = realizations_msg_list[iens];
        if (stringlist_get_size(msg)) {
            int msg_count = 0;
            for (; msg_count < stringlist_get_size(msg); ++msg_count)
                fprintf(stderr,
                        "** Warning: Function %s : Load of realization number "
                        "%d returned the following warning: %s\n",
                        __func__, iens, stringlist_iget(msg, msg_count));
        }
        stringlist_free(msg);
    }

    free(realizations_msg_list);
    bool_vector_free(iactive);
    return NULL;
}

// Internal workflow job
extern "C" C_USED void *
enkf_main_load_results_JOB(void *self, const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    int iter = 0;
    {
        const model_config_type *model_config =
            enkf_main_get_model_config(enkf_main);
        if (model_config_runpath_requires_iter(model_config))
            fprintf(
                stderr,
                "**Warning: the runpath format:%s requires an iteration number "
                "- using default:0. Use the job: LOAD_RESULT_ITER instead.\n",
                model_config_get_runpath_as_char(model_config));
    }
    return enkf_main_load_results_JOB__(enkf_main, iter, args);
}

// Internal Workflow job
extern "C" C_USED void *
enkf_main_load_results_iter_JOB(void *self, const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    stringlist_type *iens_args = stringlist_alloc_new();
    int iter;

    for (int i = 1; i < stringlist_get_size(args); i++)
        stringlist_append_copy(iens_args, stringlist_iget(args, i));

    util_sscanf_int(stringlist_iget(args, 0), &iter);
    enkf_main_load_results_JOB__(enkf_main, iter, iens_args);
    stringlist_free(iens_args);

    return NULL;
}

static void enkf_main_jobs_export_field(const enkf_main_type *enkf_main,
                                        const stringlist_type *args,
                                        field_file_format_type file_type) {
    const char *field = stringlist_iget(args, 0);
    const char *file_name = stringlist_iget(args, 1);
    int report_step = 0;
    util_sscanf_int(stringlist_iget(args, 2), &report_step);

    bool_vector_type *iactive =
        alloc_iactive_vector_from_range(args, 4, stringlist_get_size(args),
                                        enkf_main_get_ensemble_size(enkf_main));
    enkf_main_export_field(enkf_main, field, file_name, iactive, file_type,
                           report_step);
    bool_vector_free(iactive);
}

// Internal workflow job
extern "C" C_USED void *
enkf_main_export_field_JOB(void *self, const stringlist_type *args) {
    const char *file_name = stringlist_iget(args, 1);
    field_file_format_type file_type =
        field_config_default_export_format(file_name);

    if ((RMS_ROFF_FILE == file_type) || (ECL_GRDECL_FILE == file_type)) {
        enkf_main_type *enkf_main = enkf_main_safe_cast(self);
        enkf_main_jobs_export_field(enkf_main, args, file_type);
    } else
        printf("EXPORT_FIELD filename argument: File extension must be either "
               ".roff or .grdecl\n");

    return NULL;
}

// Internal workflow job
extern "C" C_USED void *
enkf_main_export_field_to_RMS_JOB(void *self, const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    enkf_main_jobs_export_field(enkf_main, args, RMS_ROFF_FILE);
    return NULL;
}

// Internal workflow job
extern "C" C_USED void *
enkf_main_export_field_to_ECL_JOB(void *self, const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    enkf_main_jobs_export_field(enkf_main, args, ECL_GRDECL_FILE);
    return NULL;
}

// Internal workflow job
extern "C" C_USED void *
enkf_main_init_misfit_table_JOB(void *self, const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    int history_length = enkf_main_get_history_length(enkf_main);
    enkf_obs_type *enkf_obs = enkf_main_get_obs(enkf_main);
    int ens_size = enkf_main_get_ensemble_size(enkf_main);
    enkf_fs_type *fs = enkf_main_job_get_fs(enkf_main);
    bool force_update = true;
    const ensemble_config_type *ensemble_config =
        enkf_main_get_ensemble_config(enkf_main);

    misfit_ensemble_type *misfit_ensemble = enkf_fs_get_misfit_ensemble(fs);
    misfit_ensemble_initialize(misfit_ensemble, ensemble_config, enkf_obs, fs,
                               ens_size, history_length, force_update);

    return NULL;
}

static void enkf_main_export_runpath_file(enkf_main_type *enkf_main,
                                          const int_vector_type *realizations,
                                          const int_vector_type *iterations) {

    const model_config_type *model_config =
        enkf_main_get_model_config(enkf_main);
    const char *basename_fmt = model_config_get_jobname_fmt(model_config);
    const char *runpath_fmt = model_config_get_runpath_as_char(model_config);
    const hook_manager_type *hook_manager =
        enkf_main_get_hook_manager(enkf_main);

    runpath_list_type *runpath_list =
        runpath_list_alloc(hook_manager_get_runpath_list_file(hook_manager));

    for (int iter = 0; iter < int_vector_size(iterations); ++iter) {
        for (int iens = 0; iens < int_vector_size(realizations); ++iens) {
            int iter_value = int_vector_iget(iterations, iter);
            int iens_value = int_vector_iget(realizations, iens);
            char *basename;
            char *runpath;

            if (basename_fmt)
                basename = util_alloc_sprintf(basename_fmt, iens_value);
            else
                basename = util_alloc_sprintf("--%d", iens_value);

            if (model_config_runpath_requires_iter(model_config))
                runpath =
                    util_alloc_sprintf(runpath_fmt, iens_value, iter_value);
            else
                runpath = util_alloc_sprintf(runpath_fmt, iens_value);

            runpath_list_add(runpath_list, iens_value, iter_value, runpath,
                             basename);

            free(basename);
            free(runpath);
        }
    }
    runpath_list_fprintf(runpath_list);
    runpath_list_free(runpath_list);
}

// Internal workflow job
extern "C" C_USED void *
enkf_main_export_runpath_file_JOB(void *self, const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    int ensemble_size = enkf_main_get_ensemble_size(enkf_main);
    const analysis_config_type *analysis_config =
        enkf_main_get_analysis_config(enkf_main);
    analysis_iter_config_type *iter_config =
        analysis_config_get_iter_config(analysis_config);
    int num_iterations = analysis_iter_config_get_num_iterations(iter_config);
    const model_config_type *model_config =
        enkf_main_get_model_config(enkf_main);
    int_vector_type *realizations = int_vector_alloc(1, 0);
    int_vector_init_range(realizations, 0, ensemble_size, 1);
    int_vector_type *iterations = int_vector_alloc(1, 0);

    if (stringlist_get_size(args) > 0) {
        int offset = 0;
        while (true) {
            if (offset == stringlist_get_size(args))
                break;
            if (0 == strcmp("|", stringlist_iget(args, offset)))
                break;
            ++offset;
        }

        if (0 != strcmp("*", stringlist_iget(args, 0))) {
            char *range_str =
                stringlist_alloc_joined_substring(args, 0, offset, "");
            string_util_init_value_list(range_str, realizations);
            free(range_str);
        }

        if ((offset < stringlist_get_size(args)) &&
            model_config_runpath_requires_iter(model_config)) {
            if (0 == strcmp("*", stringlist_iget(args, (offset + 1))))
                int_vector_init_range(iterations, 0, num_iterations, 1);
            else {
                char *range_str = stringlist_alloc_joined_substring(
                    args, offset + 1, stringlist_get_size(args), "");
                string_util_init_value_list(range_str, iterations);
                free(range_str);
            }
        }
    }

    enkf_main_export_runpath_file(enkf_main, realizations, iterations);

    int_vector_free(realizations);
    int_vector_free(iterations);

    return NULL;
}

// Internal workflow job
extern "C" C_USED void *
enkf_main_analysis_update_JOB(void *self, const stringlist_type *args) {
    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    enkf_fs_type *source_fs =
        enkf_main_mount_alt_fs(enkf_main, stringlist_iget(args, 0), false);
    enkf_fs_type *target_fs =
        enkf_main_mount_alt_fs(enkf_main, stringlist_iget(args, 1), true);

    enkf_main_smoother_update(enkf_main, source_fs, target_fs);

    enkf_fs_decref(source_fs);
    enkf_fs_decref(target_fs);
    return NULL;
}

// Internal workflow job
extern "C" C_USED void *
enkf_main_pre_simulation_copy_JOB(void *self, const stringlist_type *args) {
    const char *source_path = stringlist_iget(args, 0);

    if (!util_entry_exists(source_path)) {
        logger->error("Error in workflow job PRE_SIMULATION_COPY"
                      "- source argument: {} not existing\n",
                      source_path);
        return NULL;
    }

    enkf_main_type *enkf_main = enkf_main_safe_cast(self);
    model_config_type *model_config = enkf_main_get_model_config(enkf_main);
    if (!model_config_data_root_is_set(model_config)) {
        logger->error(
            "Error in workflow job PRE_SIMULATION_COPY DATA_ROOT not set");
        return NULL;
    }

    char *target_path;
    if (stringlist_get_size(args) == 2) {
        const char *arg_path = stringlist_iget(args, 1);
        target_path = util_alloc_filename(
            model_config_get_data_root(model_config), arg_path, NULL);
    } else
        target_path =
            util_alloc_string_copy(model_config_get_data_root(model_config));

    util_make_path(target_path);
    if (util_is_directory(source_path)) {
        util_copy_directory(source_path, target_path);
        logger->info("Copying directory {} -> {}", source_path, target_path);
    } else {
        char *base_name = util_split_alloc_filename(source_path);
        char *target_file = util_alloc_filename(target_path, base_name, NULL);

        util_copy_file(source_path, target_file);
        logger->info("Copying file {} -> {}", source_path, target_path);

        free(base_name);
        free(target_file);
    }

    free(target_path);
    return NULL;
}
