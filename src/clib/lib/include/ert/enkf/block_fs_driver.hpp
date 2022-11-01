#ifndef ERT_BLOCK_FS_DRIVER_H
#define ERT_BLOCK_FS_DRIVER_H

#include <stdbool.h>
#include <stdio.h>

#include <ert/enkf/fs_types.hpp>

typedef struct buffer_struct buffer_type;
typedef struct bfs_config_struct bfs_config_type;
typedef struct bfs_struct bfs_type;

namespace ert {

class block_fs_driver {
    int num_fs;
    bfs_config_type *config{};
    bfs_type **fs_list;

public:
    block_fs_driver(int num_fs);
    ~block_fs_driver();

    static block_fs_driver *new_(bool read_only, int num_fs,
                                 const char *mountfile_fmt);
    static block_fs_driver *open(FILE *fstab_stream, const char *mount_point,
                                 bool read_only);

    bool has_node(const char *node_key, int report_step, int iens);
    void load_node(const char *node_key, int report_step, int iens,
                   buffer_type *buffer);
    void save_node(const char *node_key, int report_step, int iens,
                   buffer_type *buffer);
    void save_node(const char *node_key, int report_step, int iens,
                   const void *ptr, size_t data_size);

    bool has_vector(const char *node_key, int iens);
    void load_vector(const char *node_key, int iens, buffer_type *buffer);
    void save_vector(const char *node_key, int iens, buffer_type *buffer);

    void fsync();

private:
    void mount();
    bfs_type *get_fs(int iens);
};

} // namespace ert
void block_fs_driver_create_fs(FILE *stream, const char *mount_point,
                               fs_driver_enum driver_type, int num_fs,
                               const char *ens_path_fmt, const char *filename);
void block_fs_driver_fskip(FILE *fstab_stream);

#endif
