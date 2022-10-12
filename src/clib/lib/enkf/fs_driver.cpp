#include <filesystem>

#include <ert/util/util.h>

#include <ert/enkf/fs_driver.hpp>
#include <ert/enkf/fs_types.hpp>

namespace fs = std::filesystem;

/*
   The underlying base types (abstract - with no accompanying
   implementation); these two type ID's are not exported outside this
   file. They are not stored to disk, and only used in an attempt
   yo verify run-time casts.
*/
#define FS_DRIVER_ID 10

void fs_driver_init(fs_driver_type *driver) {
    driver->type_id = FS_DRIVER_ID;

    driver->load_node = NULL;
    driver->save_node = NULL;
    driver->has_node = NULL;

    driver->load_vector = NULL;
    driver->save_vector = NULL;
    driver->has_vector = NULL;

    driver->free_driver = NULL;
    driver->fsync_driver = NULL;
}

void fs_driver_init_fstab(FILE *stream, fs_driver_impl driver_id) {
    util_fwrite_long(FS_MAGIC_ID, stream);
    util_fwrite_int(CURRENT_FS_VERSION, stream);
    util_fwrite_int(driver_id, stream);
}

char *fs_driver_alloc_fstab_file(const char *path) {
    return util_alloc_filename(path, "ert_fstab", NULL);
}

/**
   Will open fstab stream and return it. The semantics with respect to
   existing/not existnig fstab file depends on the value of the
   create parameter:

   @param path The file path
   @param create If create=true and the fstab file exists the function will
       return NULL, otherwise it will return a stream opened for writing to the
       fstab file.

       if create = False and the fstab file exists the the function will return a
       stream opened for reading of the fstab file, otherwise it will return
       NULL.

   @return A stream to the opened file.
*/
FILE *fs_driver_open_fstab(const char *path, bool create) {
    FILE *stream = NULL;
    char *fstab_file = fs_driver_alloc_fstab_file(path);
    if (create)
        util_make_path(path);

    if (fs::exists(fstab_file) != create) {
        if (create)
            stream = util_fopen(fstab_file, "w");
        else
            stream = util_fopen(fstab_file, "r");
    }
    free(fstab_file);
    return stream;
}

void fs_driver_assert_magic(FILE *stream) {
    long fs_magic = util_fread_long(stream);
    if (fs_magic != FS_MAGIC_ID)
        util_abort("%s: Fstab magic marker incorrect \n", __func__);
}

void fs_driver_assert_version(FILE *stream, const char *mount_point) {
    int file_version = util_fread_int(stream);

    if (file_version > CURRENT_FS_VERSION)
        util_exit("%s: The file system you are trying to access was "
                  "created with a newer version of ERT.\n",
                  __func__);

    if (file_version < CURRENT_FS_VERSION) {
        fprintf(stderr,
                "----------------------------------------------------------"
                "-------------------------------------------\n");
        fprintf(stderr,
                "  %s: The file system you are trying to access was "
                "created with an old version of ERT.\n",
                __func__);
        fprintf(stderr, "  ert_fs_version: %d \n", CURRENT_FS_VERSION);
        fprintf(stderr, "  %s version: %d \n", mount_point, file_version);

        util_exit("  EXIT\n");
        fprintf(stderr,
                "----------------------------------------------------------"
                "-------------------------------------------\n");
    }
}

int fs_driver_fread_version(FILE *stream) {
    long fs_magic = util_fread_long(stream);
    if (fs_magic != FS_MAGIC_ID)
        return -1;
    else {
        int file_version = util_fread_int(stream);
        return file_version;
    }
}
