#include <stdlib.h>

#include <ert/enkf/run_arg.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>

void call_get_queue_index(void *arg) {
    auto run_arg = static_cast<run_arg_type *>(arg);
    run_arg_get_queue_index(run_arg);
}

void call_set_queue_index(void *arg) {
    auto run_arg = static_cast<run_arg_type *>(arg);
    run_arg_set_queue_index(run_arg, 88);
}

void test_queue_index() {
    ecl::util::TestArea ta("queue_index");
    {
        enkf_fs_type *fs =
            enkf_fs_create_fs("sim", BLOCK_FS_DRIVER_ID, 1, true);
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
    auto run_arg = static_cast<run_arg_type *>(arg);
    run_arg_get_sim_fs(run_arg);
}

void call_get_update_target_fs(void *arg) {
    auto run_arg = static_cast<run_arg_type *>(arg);
}

void test_SMOOTHER_RUN() {
    ecl::util::TestArea ta("smoother");
    {
        enkf_fs_type *sim_fs =
            enkf_fs_create_fs("sim", BLOCK_FS_DRIVER_ID, 1, true);
        enkf_fs_type *target_fs =
            enkf_fs_create_fs("target", BLOCK_FS_DRIVER_ID, 1, true);
        run_arg_type *run_arg =
            run_arg_alloc("run_id", sim_fs, 0, 6, "path", "BASE");
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
            enkf_fs_create_fs("sim", BLOCK_FS_DRIVER_ID, 1, true);

        run_arg_type *run_arg =
            run_arg_alloc("run_id", init_fs, 0, 6, "path", NULL);
        test_assert_ptr_equal(run_arg_get_sim_fs(run_arg), init_fs);

        run_arg_free(run_arg);
        enkf_fs_umount(init_fs);
    }
}

void test_ENSEMBLE_EXPERIMENT() {
    ecl::util::TestArea ta("ens");
    {
        enkf_fs_type *fs =
            enkf_fs_create_fs("sim", BLOCK_FS_DRIVER_ID, 1, true);

        run_arg_type *run_arg =
            run_arg_alloc("run_id", fs, 0, 6, "path", "BASE");

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
