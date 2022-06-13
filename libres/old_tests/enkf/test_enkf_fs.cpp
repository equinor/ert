/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_fs.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <ert/enkf/block_fs_driver.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>

namespace fs = std::filesystem;

typedef struct {
    pthread_mutex_t mutex1;
    pthread_mutex_t mutex2;
} shared_data;

static shared_data *data = NULL;

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
    test_assert_NULL(enkf_fs_create_fs("mnt", BLOCK_FS_DRIVER_ID, false));
    test_assert_true(enkf_fs_exists("mnt"));
    {
        enkf_fs_type *fs = enkf_fs_mount("mnt");
        test_assert_true(fs::exists("mnt/mnt.lock"));
        test_assert_true(enkf_fs_is_instance(fs));
        enkf_fs_decref(fs);
        test_assert_false(fs::exists("mnt/mnt.lock"));
    }
    {
        enkf_fs_type *fs = enkf_fs_create_fs("mnt2", BLOCK_FS_DRIVER_ID, true);
        test_assert_true(enkf_fs_is_instance(fs));
        enkf_fs_decref(fs);
    }
}

void test_refcount() {
    ecl::util::TestArea ta("ref_count");
    enkf_fs_create_fs("mnt", BLOCK_FS_DRIVER_ID, false);
    {
        enkf_fs_type *fs = enkf_fs_mount("mnt");
        test_assert_int_equal(1, enkf_fs_get_refcount(fs));
        enkf_fs_decref(fs);
    }
}

void createFS() {

    pthread_mutex_lock(&data->mutex1);
    pid_t pid = fork();

    if (pid == 0) {
        enkf_fs_type *fs_false = enkf_fs_mount("mnt");
        test_assert_false(enkf_fs_is_read_only(fs_false));
        test_assert_true(fs::exists("mnt/mnt.lock"));
        pthread_mutex_unlock(&data->mutex1);
        pthread_mutex_lock(&data->mutex2);
        enkf_fs_decref(fs_false);
        pthread_mutex_unlock(&data->mutex2);
        exit(0);
    }
}

void test_fwrite_readonly(void *arg) {
    enkf_fs_type *fs = enkf_fs_safe_cast(arg);
    /*
     The arguments here are completely bogus; the important thing is
     that this fwrite call should be intercepted by a util_abort()
     call (which is again intercepted by the testing function) before
     the argument are actually accessed.
  */
    enkf_fs_fwrite_node(fs, NULL, "KEY", PARAMETER, 100, 1);
}

void initialise_shared() {
    // place our shared data in shared memory
    int prot = PROT_READ | PROT_WRITE;
#ifdef __linux
    int flags = MAP_SHARED | MAP_ANONYMOUS;
#elif __APPLE__
    int flags = MAP_SHARED | MAP_ANON;
#endif

    data = (shared_data *)mmap(NULL, sizeof(shared_data), prot, flags, -1, 0);
    assert(data);

    // initialise mutex so it works properly in shared memory
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&data->mutex1, &attr);
    pthread_mutex_init(&data->mutex2, &attr);
}

/*
  This test needs to fork off a seperate process to test the cross-process file locking.
*/
void test_read_only2() {
    initialise_shared();
    {
        ecl::util::TestArea ta("ro2");
        enkf_fs_create_fs("mnt", BLOCK_FS_DRIVER_ID, false);
        pthread_mutex_lock(&data->mutex2);
        createFS();
        pthread_mutex_lock(&data->mutex1);
        {
            enkf_fs_type *fs_false = enkf_fs_mount("mnt");
            test_assert_true(enkf_fs_is_read_only(fs_false));
            test_assert_util_abort("enkf_fs_fwrite_node", test_fwrite_readonly,
                                   fs_false);
            enkf_fs_decref(fs_false);
        }
        pthread_mutex_unlock(&data->mutex2);
        pthread_mutex_unlock(&data->mutex1);
        pthread_mutex_lock(&data->mutex2);
    }
    pthread_mutex_unlock(&data->mutex2);
    munmap(data, sizeof(data));
}

int main(int argc, char **argv) {
    test_mount();
    test_refcount();
    test_read_only2();
    test_block_fs_driver_create_fs();
    exit(0);
}
