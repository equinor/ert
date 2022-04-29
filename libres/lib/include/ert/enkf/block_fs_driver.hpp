/*
   Copyright (C) 2011-2021  Equinor ASA, Norway.

   The file 'block_fs_driver.h' is part of ERT - Ensemble based Reservoir Tool.

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

    bool has_vector(const char *node_key, int iens);
    void load_vector(const char *node_key, int iens, buffer_type *buffer);
    void save_vector(const char *node_key, int iens, buffer_type *buffer);

    void fsync();

private:
    void mount();
    bfs_type *get_fs(int iens);
};

} // namespace ert

bool block_fs_sscanf_key(const char *key, char **config_key, int *__report_step,
                         int *__iens);
void block_fs_driver_create_fs(FILE *stream, const char *mount_point,
                               fs_driver_enum driver_type, int num_fs,
                               const char *ens_path_fmt, const char *filename);
void block_fs_driver_fskip(FILE *fstab_stream);

#endif
