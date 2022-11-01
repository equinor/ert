#include <cstdio>
#include <filesystem>
#include <future>
#include <vector>

namespace fs = std::filesystem;

#include <stdio.h>
#include <stdlib.h>

#include <ert/util/buffer.h>
#include <ert/util/util.h>

#include <ert/res_util/block_fs.hpp>

#include <ert/enkf/block_fs_driver.hpp>
#include <ert/enkf/fs_types.hpp>

#include <fmt/format.h>

typedef struct bfs_struct bfs_type;
typedef struct bfs_config_struct bfs_config_type;

struct bfs_config_struct {
    int fsync_interval;
    bool read_only;
};

struct bfs_struct {
    block_fs_type *block_fs;
    /** The full path to the file mounted by the block_fs layer - including extension.*/
    char *mountfile;

    const bfs_config_type *config;
};

bfs_config_type *bfs_config_alloc(bool read_only) {
    const int fsync_interval =
        10; /* An fsync() call is issued for every 10'th write. */

    {
        bfs_config_type *config =
            (bfs_config_type *)util_malloc(sizeof *config);
        config->fsync_interval = fsync_interval;
        config->read_only = read_only;
        return config;
    }
}

void bfs_config_free(bfs_config_type *config) { free(config); }

static void bfs_close(bfs_type *bfs) {
    if (bfs->block_fs != NULL)
        block_fs_close(bfs->block_fs);
    free(bfs->mountfile);
    free(bfs);
}

static bfs_type *bfs_alloc(const bfs_config_type *config) {
    bfs_type *fs = (bfs_type *)util_malloc(sizeof *fs);
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
    bfs->block_fs = block_fs_mount(bfs->mountfile, config->fsync_interval,
                                   config->read_only);
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

void ert::block_fs_driver::save_node(const char *node_key, int report_step, int iens,
                                     const void *ptr, size_t data_size) {
    auto key = fmt::format("{}.{}.{}", node_key, report_step, iens);
    bfs_type *bfs = this->get_fs(iens);
    block_fs_fwrite_file(bfs->block_fs, key.c_str(), ptr, data_size);
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
                                                 const char *mountfile_fmt) {
    ert::block_fs_driver *driver = new ert::block_fs_driver(num_fs);
    driver->config = bfs_config_alloc(read_only);
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

/**
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

    ert::block_fs_driver *driver =
        ert::block_fs_driver::new_(read_only, num_fs, mountfile_fmt);

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
