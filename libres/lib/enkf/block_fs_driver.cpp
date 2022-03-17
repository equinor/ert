/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'block_fs_driver.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <cstdio>
#include <future>
#include <vector>

namespace fs = std::filesystem;

#include <stdlib.h>
#include <stdio.h>

#include <ert/util/util.h>
#include <ert/util/buffer.h>

#include <ert/res_util/block_fs.hpp>

#include <ert/enkf/fs_types.hpp>
#include <ert/enkf/block_fs_driver.hpp>

typedef struct bfs_struct bfs_type;
typedef struct bfs_config_struct bfs_config_type;

struct bfs_config_struct {
    bool read_only;
    bool bfs_lock;
};

#define BFS_TYPE_ID 5510643

struct bfs_struct {
    UTIL_TYPE_ID_DECLARATION;
    block_fs_type *block_fs;
    char *
        mountfile; // The full path to the file mounted by the block_fs layer - including extension.

    const bfs_config_type *config;
};

bfs_config_type *bfs_config_alloc(bool read_only, bool bfs_lock) {
    {
        bfs_config_type *config =
            (bfs_config_type *)util_malloc(sizeof *config);
        config->read_only = read_only;
        config->bfs_lock = bfs_lock;
        return config;
    }
}

void bfs_config_free(bfs_config_type *config) { free(config); }

static UTIL_SAFE_CAST_FUNCTION(bfs, BFS_TYPE_ID);

static void bfs_close(bfs_type *bfs) {
    if (bfs->block_fs != NULL)
        block_fs_close(bfs->block_fs, false);
    free(bfs->mountfile);
    free(bfs);
}

static bfs_type *bfs_alloc(const bfs_config_type *config) {
    bfs_type *fs = (bfs_type *)util_malloc(sizeof *fs);
    UTIL_TYPE_ID_INIT(fs, BFS_TYPE_ID);
    fs->config = config;

    // New init
    fs->mountfile = NULL;

    return fs;
}

static bfs_type *bfs_alloc_new(const bfs_config_type *config, char *mountfile) {
    bfs_type *bfs = bfs_alloc(config);

    bfs->mountfile =
        mountfile; // Warning pattern break: This is allocated in external scope; and the bfs takes ownership.
    return bfs;
}

static void bfs_mount(bfs_type *bfs) {
    const bfs_config_type *config = bfs->config;
    bfs->block_fs =
        block_fs_mount(bfs->mountfile, config->read_only, config->bfs_lock);
}

static void bfs_fsync(bfs_type *bfs) { block_fs_fsync(bfs->block_fs); }

static char *block_fs_driver_alloc_node_key(const char *node_key,
                                            int report_step, int iens) {
    char *key = util_alloc_sprintf("%s.%d.%d", node_key, report_step, iens);
    return key;
}

static char *block_fs_driver_alloc_vector_key(const char *node_key, int iens) {
    char *key = util_alloc_sprintf("%s.%d", node_key, iens);
    return key;
}

/*
   This function will take an input string, and try to to parse it as
   string.int.int, where string is the normal enkf key, and the two
   integers are report_step and ensemble number respectively. The
   storage for the enkf_key is allocated here in this function, and
   must be freed by the calling scope.

   If the parsing fails the function will return false, and *config_key
   will be set to NULL; in this case the report_step and iens poinyers
   will not be touched.
*/

bool block_fs_sscanf_key(const char *key, char **config_key, int *__report_step,
                         int *__iens) {
    char **tmp;
    int num_items;

    *config_key = NULL;
    util_split_string(
        key, ".", &num_items,
        &tmp); /* The key can contain additional '.' - can not use sscanf(). */
    if (num_items >= 3) {
        int report_step, iens;
        if (util_sscanf_int(tmp[num_items - 2], &report_step) &&
            util_sscanf_int(tmp[num_items - 1], &iens)) {
            /* OK - all is hunkadory */
            *__report_step = report_step;
            *__iens = iens;
            *config_key = util_alloc_joined_string(
                (const char **)tmp, num_items - 2,
                "."); /* This must bee freed by the calling scope */
            util_free_stringlist(tmp, num_items);
            return true;
        } else
            /* Failed to parse the two last items as integers. */
            return false;
    } else
        /* Did not have at least three items. */
        return false;
}

bfs_type *ert::block_fs_driver::get_fs(int iens) {
    int phase = (iens % this->num_fs);
    return this->fs_list[phase];
}

void ert::block_fs_driver::load_node(const char *node_key, int report_step,
                                     int iens, buffer_type *buffer) {
    char *key = block_fs_driver_alloc_node_key(node_key, report_step, iens);
    bfs_type *bfs = this->get_fs(iens);

    block_fs_fread_realloc_buffer(bfs->block_fs, key, buffer);

    free(key);
}

void ert::block_fs_driver::load_vector(const char *node_key, int iens,
                                       buffer_type *buffer) {
    char *key = block_fs_driver_alloc_vector_key(node_key, iens);
    bfs_type *bfs = this->get_fs(iens);

    block_fs_fread_realloc_buffer(bfs->block_fs, key, buffer);
    free(key);
}

void ert::block_fs_driver::save_node(const char *node_key, int report_step,
                                     int iens, buffer_type *buffer) {
    char *key = block_fs_driver_alloc_node_key(node_key, report_step, iens);
    bfs_type *bfs = this->get_fs(iens);
    block_fs_fwrite_buffer(bfs->block_fs, key, buffer);
    free(key);
}

void ert::block_fs_driver::save_vector(const char *node_key, int iens,
                                       buffer_type *buffer) {
    char *key = block_fs_driver_alloc_vector_key(node_key, iens);
    bfs_type *bfs = this->get_fs(iens);
    block_fs_fwrite_buffer(bfs->block_fs, key, buffer);
    free(key);
}

bool ert::block_fs_driver::has_node(const char *node_key, int report_step,
                                    int iens) {
    char *key = block_fs_driver_alloc_node_key(node_key, report_step, iens);
    bfs_type *bfs = this->get_fs(iens);
    bool has_node = block_fs_has_file(bfs->block_fs, key);
    free(key);
    return has_node;
}

bool ert::block_fs_driver::has_vector(const char *node_key, int iens) {
    char *key = block_fs_driver_alloc_vector_key(node_key, iens);
    bfs_type *bfs = this->get_fs(iens);
    bool has_node = block_fs_has_file(bfs->block_fs, key);
    free(key);
    return has_node;
}

ert::block_fs_driver::~block_fs_driver() {
    // Sometimes only one is managed, so no need to spin up parallelism
    if (this->num_fs == 1) {
        bfs_close(this->fs_list[0]);
    } else {
        std::vector<std::future<void>> futures;
        for (int driver_nr = 0; driver_nr < this->num_fs; ++driver_nr)
            futures.push_back(std::async(std::launch::async, bfs_close,
                                         this->fs_list[driver_nr]));

        // Wait for all futures to finish
        for (auto &fut : futures)
            fut.get();
    }

    bfs_config_free(this->config);
    free(this->fs_list);
}

void ert::block_fs_driver::fsync() {
    int driver_nr;
    for (driver_nr = 0; driver_nr < this->num_fs; driver_nr++)
        bfs_fsync(this->fs_list[driver_nr]);
}

ert::block_fs_driver::block_fs_driver(int num_fs) : num_fs(num_fs) {
    this->fs_list = (bfs_type **)util_calloc(this->num_fs, sizeof(bfs_type *));
}

ert::block_fs_driver *ert::block_fs_driver::new_(bool read_only, int num_fs,
                                                 const char *mountfile_fmt,
                                                 bool block_level_lock) {
    ert::block_fs_driver *driver = new ert::block_fs_driver(num_fs);
    driver->config = bfs_config_alloc(read_only, block_level_lock);
    {
        for (int ifs = 0; ifs < driver->num_fs; ifs++)
            driver->fs_list[ifs] = bfs_alloc_new(
                driver->config, util_alloc_sprintf(mountfile_fmt, ifs));
    }
    return driver;
}

void ert::block_fs_driver::mount() {
    // Sometimes only one is managed, so no need to spin up parallelism
    if (this->num_fs == 1) {
        bfs_mount(this->fs_list[0]);
    } else {
        std::vector<std::future<void>> futures;
        for (int driver_nr = 0; driver_nr < this->num_fs; ++driver_nr)
            futures.push_back(std::async([](bfs_type *bfs) { bfs_mount(bfs); },
                                         this->fs_list[driver_nr]));

        // Wait for all futures to finish
        for (auto &fut : futures)
            fut.get();
    }
}

void block_fs_driver_create_fs(FILE *stream, const char *mount_point,
                               fs_driver_enum driver_type, int num_fs,
                               const char *ens_path_fmt, const char *filename) {

    std::fwrite(&driver_type, sizeof driver_type, 1, stream);
    std::fwrite(&num_fs, sizeof num_fs, 1, stream);

    {
        std::string mountfile_fmt = std::string(ens_path_fmt) +
                                    std::string(1, UTIL_PATH_SEP_CHAR) +
                                    std::string(filename) + std::string(".mnt");

        int len = mountfile_fmt.length();
        std::fwrite(&len, sizeof len, 1, stream);
        std::fwrite(mountfile_fmt.c_str(), sizeof(char), len + 1, stream);
    }

    for (int ifs = 0; ifs < num_fs; ifs++) {
        char *path_fmt;
        asprintf(&path_fmt, ens_path_fmt, ifs);

        fs::path ens_path = fs::path(mount_point) / fs::path(path_fmt);
        fs::create_directories(ens_path);

        free(path_fmt);
    }
}

/*
  @path should contain both elements called root_path and case_path in
  the block_fs_driver_create() function.
*/

ert::block_fs_driver *ert::block_fs_driver::open(FILE *fstab_stream,
                                                 const char *mount_point,
                                                 bool read_only) {
    int num_fs = util_fread_int(fstab_stream);
    char *tmp_fmt = util_fread_alloc_string(fstab_stream);
    char *mountfile_fmt =
        util_alloc_sprintf("%s%c%s", mount_point, UTIL_PATH_SEP_CHAR, tmp_fmt);
    const bool block_level_lock = false;

    ert::block_fs_driver *driver = ert::block_fs_driver::new_(
        read_only, num_fs, mountfile_fmt, block_level_lock);

    driver->mount();

    free(tmp_fmt);
    free(mountfile_fmt);
    return driver;
}

void block_fs_driver_fskip(FILE *fstab_stream) {
    util_fskip_int(fstab_stream);
    {
        char *tmp_fmt = util_fread_alloc_string(fstab_stream);
        free(tmp_fmt);
    }
}
