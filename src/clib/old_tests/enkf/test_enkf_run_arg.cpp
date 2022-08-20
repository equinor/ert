/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'ert_run_context.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdlib.h>

#include <ert/enkf/run_arg.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>

void call_get_queue_index(void *arg) {
    run_arg_type *run_arg = run_arg_safe_cast(arg);
    run_arg_get_queue_index(run_arg);
}

void call_set_queue_index(void *arg) {
    run_arg_type *run_arg = run_arg_safe_cast(arg);
    run_arg_set_queue_index(run_arg, 88);
}

void test_queue_index() {
    ecl::util::TestArea ta("queue_index");
    {
        enkf_fs_type *fs = enkf_fs_create_fs("sim", BLOCK_FS_DRIVER_ID, true);
        run_arg_type *run_arg =
            run_arg_alloc("run_id", fs, 0, 6, "path", "base");

        test_assert_false(run_arg_is_submitted(run_arg));
        test_assert_util_abort("run_arg_get_queue_index", call_get_queue_index,
                               run_arg);

        int qi = run_arg_get_queue_index_safe(run_arg);
        test_assert_int_equal(-1, qi); // not submitted: index == -1

        run_arg_set_queue_index(run_arg, 78);
        test_assert_true(run_arg_is_submitted(run_arg));
        test_assert_int_equal(78, run_arg_get_queue_index(run_arg));

        test_assert_util_abort("run_arg_set_queue_index", call_set_queue_index,
                               run_arg);
        run_arg_free(run_arg);
        enkf_fs_umount(fs);
    }
}

void call_get_sim_fs(void *arg) {
    run_arg_type *run_arg = run_arg_safe_cast(arg);
    run_arg_get_sim_fs(run_arg);
}

void call_get_update_target_fs(void *arg) {
    run_arg_type *run_arg = run_arg_safe_cast(arg);
}

void test_SMOOTHER_RUN() {
    ecl::util::TestArea ta("smoother");
    {
        enkf_fs_type *sim_fs =
            enkf_fs_create_fs("sim", BLOCK_FS_DRIVER_ID, true);
        enkf_fs_type *target_fs =
            enkf_fs_create_fs("target", BLOCK_FS_DRIVER_ID, true);
        run_arg_type *run_arg =
            run_arg_alloc("run_id", sim_fs, 0, 6, "path", "BASE");
        test_assert_true(run_arg_is_instance(run_arg));
        test_assert_ptr_equal(run_arg_get_sim_fs(run_arg), sim_fs);
        run_arg_free(run_arg);
        enkf_fs_umount(sim_fs);
        enkf_fs_umount(target_fs);
    }
}

void test_INIT_ONLY() {
    ecl::util::TestArea ta("INIT");
    {
        enkf_fs_type *init_fs =
            enkf_fs_create_fs("sim", BLOCK_FS_DRIVER_ID, true);

        run_arg_type *run_arg =
            run_arg_alloc("run_id", init_fs, 0, 6, "path", NULL);
        test_assert_true(run_arg_is_instance(run_arg));
        test_assert_ptr_equal(run_arg_get_sim_fs(run_arg), init_fs);

        run_arg_free(run_arg);
        enkf_fs_umount(init_fs);
    }
}

void test_ENSEMBLE_EXPERIMENT() {
    ecl::util::TestArea ta("ens");
    {
        enkf_fs_type *fs = enkf_fs_create_fs("sim", BLOCK_FS_DRIVER_ID, true);

        run_arg_type *run_arg =
            run_arg_alloc("run_id", fs, 0, 6, "path", "BASE");
        test_assert_true(run_arg_is_instance(run_arg));

        test_assert_ptr_equal(run_arg_get_sim_fs(run_arg), fs);

        test_assert_string_equal(run_arg_get_run_id(run_arg), "run_id");
        run_arg_free(run_arg);
        enkf_fs_umount(fs);
    }
}

int main(int argc, char **argv) {
    test_queue_index();
    test_SMOOTHER_RUN();
    test_INIT_ONLY();
    test_ENSEMBLE_EXPERIMENT();
    exit(0);
}
