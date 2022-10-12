#include <filesystem>

#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <ert/enkf/block_fs_driver.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>

namespace fs = std::filesystem;

void test_block_fs_driver_create_fs() {
    ecl::util::TestArea ta("block_fs_driver_create_fs");

    FILE *file_write = fopen("ert_fstab", "w");
    block_fs_driver_create_fs(file_write, "mnt", DRIVER_PARAMETER, 32,
                              "Ensemble/mod_%d", "PARAMETER");
    fclose(file_write);

    FILE *file_read = fopen("ert_fstab", "r");
    int read_driver_type;
    int read_num_fs;
    int read_len;
    char read_mountfile_fmt[100];
    fread(&read_driver_type, sizeof read_driver_type, 1, file_read);
    fread(&read_num_fs, sizeof read_num_fs, 1, file_read);
    fread(&read_len, sizeof read_len, 1, file_read);
    fread(&read_mountfile_fmt, sizeof(char), 100, file_read);
    fclose(file_read);

    test_assert_int_equal(read_driver_type, 1);
    test_assert_int_equal(read_num_fs, 32);
    test_assert_int_equal(read_len, 29);
    test_assert_string_equal(read_mountfile_fmt,
                             "Ensemble/mod_%d/PARAMETER.mnt");

    // `num_fs` parameter of `block_fs_driver_create_fs` specifies
    // how many sub-folders Ensemble/ should have.
    test_assert_true(fs::exists("mnt/Ensemble/mod_0"));
    test_assert_true(fs::exists("mnt/Ensemble/mod_31"));
    test_assert_false(fs::exists("mnt/Ensemble/mod_32"));
}

void test_mount() {
    ecl::util::TestArea ta("mount");

    test_assert_false(enkf_fs_exists("mnt"));
    test_assert_NULL(enkf_fs_create_fs("mnt", BLOCK_FS_DRIVER_ID, 1, false));
    test_assert_true(enkf_fs_exists("mnt"));
    {
        enkf_fs_type *fs = enkf_fs_mount("mnt", 1);
        test_assert_true(fs::exists("mnt/mnt.lock"));
        enkf_fs_umount(fs);
        test_assert_false(fs::exists("mnt/mnt.lock"));
    }
    {
        enkf_fs_type *fs =
            enkf_fs_create_fs("mnt2", BLOCK_FS_DRIVER_ID, 1, true);
        enkf_fs_umount(fs);
    }
}

void mount(void *args) { enkf_fs_type *fs2 = enkf_fs_mount("mnt", 1); }

void test_mount_filesystem_readwrite_twice() {
    ecl::util::TestArea ta("test_mount_filesystem_readwrite_twice");
    enkf_fs_create_fs("mnt", BLOCK_FS_DRIVER_ID, 1, false);
    enkf_fs_type *fs = enkf_fs_mount("mnt", 1);

    test_assert_true(fs::exists("mnt/mnt.lock"));
    test_assert_false(enkf_fs_is_read_only(fs));

    pid_t pid = fork();
    if (pid == 0) {
        test_assert_util_abort("enkf_fs_alloc_empty", mount, NULL);
    } else {
        int child_status;
        waitpid(pid, &child_status, 0);
        test_assert_true(child_status == 0);
    }
    enkf_fs_umount(fs);
}

int main(int argc, char **argv) {
    test_mount();
    test_mount_filesystem_readwrite_twice();
    test_block_fs_driver_create_fs();
    exit(0);
}
