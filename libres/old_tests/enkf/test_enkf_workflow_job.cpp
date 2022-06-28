/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_workflow_job_test.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <filesystem>

#include <stdio.h>
#include <stdlib.h>

#include <ert/enkf/ert_test_context.hpp>
#include <ert/util/test_util.h>

namespace fs = std::filesystem;

ert_test_context_type *create_context(const char *config_file,
                                      const char *name) {
    ert_test_context_type *test_context =
        ert_test_context_alloc(name, config_file);
    test_assert_not_NULL(test_context);
    return test_context;
}

void test_pre_simulation_copy__(ert_test_context_type *test_context,
                                const char *job_name, const char *source_path,
                                const char *target_path) {

    stringlist_type *args = stringlist_alloc_new();
    stringlist_append_copy(args, source_path);
    if (target_path)
        stringlist_append_copy(args, target_path);
    ert_test_context_run_worklow_job(test_context, job_name, args);

    stringlist_free(args);
}

void test_pre_simulation_copy(ert_test_context_type *test_context,
                              const char *job_name, const char *job_file) {
    enkf_main_type *enkf_main = ert_test_context_get_main(test_context);
    model_config_type *model_config = enkf_main_get_model_config(enkf_main);
    test_assert_false(model_config_data_root_is_set(model_config));

    test_assert_true(ert_test_context_install_workflow_job(test_context,
                                                           job_name, job_file));

    test_pre_simulation_copy__(test_context, job_name, "does_not_exist",
                               "target");

    {
        fs::create_directories(
            fs::path("input/path/xxx/model/file").remove_filename());
        FILE *f = fopen(fs::path("input/path/xxx/model/file").c_str(), "w");
        fprintf(f, "File \n");
        fclose(f);
    }
    test_pre_simulation_copy__(test_context, job_name, "input/path/xxx/model",
                               NULL);
    test_pre_simulation_copy__(test_context, job_name, "input/path/xxx/model",
                               "target");
    test_pre_simulation_copy__(test_context, job_name,
                               "input/path/xxx/model/file",
                               "target/extra_path");
    test_pre_simulation_copy__(test_context, job_name, "input/path/xxx/model",
                               "target/extra_path2");

    test_assert_false(util_is_file("root/model/file"));
    test_assert_false(util_is_file("root/target/model/file"));
    test_assert_false(util_is_file("root/target/extra_path/file"));
    test_assert_false(util_is_file("root/target/extra_path2/model/file"));

    model_config_set_data_root(model_config, "root");
    test_assert_true(model_config_data_root_is_set(model_config));

    test_pre_simulation_copy__(test_context, job_name, "input/path/xxx/model",
                               NULL);
    test_pre_simulation_copy__(test_context, job_name, "input/path/xxx/model",
                               "target");
    test_pre_simulation_copy__(test_context, job_name,
                               "input/path/xxx/model/file",
                               "target/extra_path");
    test_pre_simulation_copy__(test_context, job_name, "input/path/xxx/model",
                               "target/extra_path2");

    test_assert_true(util_is_file("root/model/file"));
    test_assert_true(util_is_file("root/target/model/file"));
    test_assert_true(util_is_file("root/target/extra_path/file"));
    test_assert_true(util_is_file("root/target/extra_path2/model/file"));
}

void test_create_case_job(ert_test_context_type *test_context,
                          const char *job_name, const char *job_file) {
    stringlist_type *args = stringlist_alloc_new();
    stringlist_append_copy(args, "newly_created_case");
    test_assert_true(ert_test_context_install_workflow_job(test_context,
                                                           job_name, job_file));
    test_assert_true(
        ert_test_context_run_worklow_job(test_context, job_name, args));

    char *new_case = util_alloc_filename("storage", "newly_created_case", NULL);
    test_assert_true(util_is_directory(new_case));
    free(new_case);

    stringlist_free(args);
}

void test_init_case_job(ert_test_context_type *test_context,
                        const char *job_name, const char *job_file) {
    stringlist_type *args = stringlist_alloc_new();
    enkf_main_type *enkf_main = ert_test_context_get_main(test_context);

    test_assert_true(
        ert_test_context_install_workflow_job(test_context, "JOB", job_file));

    //Test init current case from existing
    {
        enkf_fs_type *cur_fs =
            enkf_main_mount_alt_fs(enkf_main, "new_current_case", true);
        enkf_main_select_fs(enkf_main, "new_current_case");

        test_assert_ptr_not_equal(cur_fs, enkf_main_get_fs(enkf_main));

        stringlist_append_copy(args, "default"); //case to init from
        test_assert_true(
            ert_test_context_run_worklow_job(test_context, "JOB", args));

        enkf_fs_decref(cur_fs);
    }

    {
        const char *current_case = enkf_main_get_current_fs(enkf_main);
        test_assert_string_equal(current_case, "new_current_case");
        test_assert_true(enkf_fs_has_node(
            enkf_main_get_fs(enkf_main), "PERMZ", PARAMETER, 0,
            0)); // This had state = ANALYZED; might be unfixable.

        enkf_fs_type *default_fs =
            enkf_main_mount_alt_fs(enkf_main, "default", true);
        state_map_type *default_state_map = enkf_fs_get_state_map(default_fs);
        state_map_type *current_state_map =
            enkf_fs_get_state_map(enkf_main_get_fs(enkf_main));
        test_assert_int_equal(state_map_get_size(default_state_map),
                              state_map_get_size(current_state_map));
        enkf_fs_decref(default_fs);
    }

    //Test init case from existing case:
    stringlist_clear(args);
    stringlist_append_copy(args, "default"); //case to init from
    stringlist_append_copy(args, "new_not_current_case");
    test_assert_true(
        ert_test_context_run_worklow_job(test_context, "JOB", args));
    {
        enkf_fs_type *fs =
            enkf_main_mount_alt_fs(enkf_main, "new_not_current_case", true);
        test_assert_not_NULL(fs);
        test_assert_true(enkf_fs_has_node(
            fs, "PERMZ", PARAMETER, 0,
            0)); // This had state = ANALYZED; might be unfixable.

        enkf_fs_type *default_fs =
            enkf_main_mount_alt_fs(enkf_main, "default", true);
        state_map_type *default_state_map = enkf_fs_get_state_map(default_fs);
        state_map_type *new_state_map = enkf_fs_get_state_map(fs);
        test_assert_int_equal(state_map_get_size(default_state_map),
                              state_map_get_size(new_state_map));
        enkf_fs_decref(fs);
    }

    stringlist_free(args);
}

int main(int argc, const char **argv) {
    enkf_main_install_SIGNALS();

    const char *config_file = argv[1];
    const char *config_file_iterations = argv[2];
    const char *job_file_create_case = argv[3];
    const char *job_file_init_case_job = argv[4];
    const char *job_file_export_runpath = argv[6];
    const char *job_file_pre_simulation_copy = argv[7];

    ert_test_context_type *test_context =
        create_context(config_file, "enkf_workflow_job_test");
    {
        test_create_case_job(test_context, "JOB1", job_file_create_case);
        test_init_case_job(test_context, "JOB2", job_file_init_case_job);
        test_pre_simulation_copy(test_context, "JOBB",
                                 job_file_pre_simulation_copy);
    }
    ert_test_context_free(test_context);
    exit(0);
}
