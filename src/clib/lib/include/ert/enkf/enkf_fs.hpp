#ifndef ERT_ENKF_FS_H
#define ERT_ENKF_FS_H
#include <stdbool.h>

#include <ert/util/buffer.h>

#include <ert/enkf/enkf_fs_type.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/fs_driver.hpp>
#include <ert/enkf/fs_types.hpp>
#include <ert/enkf/time_map.hpp>

const char *enkf_fs_get_mount_point(const enkf_fs_type *fs);
extern "C" bool enkf_fs_is_read_only(const enkf_fs_type *fs);
extern "C" void enkf_fs_fsync(enkf_fs_type *fs);

enkf_fs_type *enkf_fs_get_ref(enkf_fs_type *fs);
extern "C" enkf_fs_type *enkf_fs_mount(const char *path, unsigned ensemble_size,
                                       bool read_only = false);
void enkf_fs_fwrite_node(enkf_fs_type *enkf_fs, buffer_type *buffer,
                         const char *node_key, enkf_var_type var_type,
                         int report_step, int iens);

void enkf_fs_fwrite_vector(enkf_fs_type *enkf_fs, buffer_type *buffer,
                           const char *node_key, enkf_var_type var_type,
                           int iens);

bool enkf_fs_exists(const char *mount_point);

extern "C" void enkf_fs_sync(enkf_fs_type *fs);

void enkf_fs_fread_node(enkf_fs_type *enkf_fs, buffer_type *buffer,
                        const char *node_key, enkf_var_type var_type,
                        int report_step, int iens);

void enkf_fs_fread_vector(enkf_fs_type *enkf_fs, buffer_type *buffer,
                          const char *node_key, enkf_var_type var_type,
                          int iens);

bool enkf_fs_has_vector(enkf_fs_type *enkf_fs, const char *node_key,
                        enkf_var_type var_type, int iens);
bool enkf_fs_has_node(enkf_fs_type *enkf_fs, const char *node_key,
                      enkf_var_type var_type, int report_step, int iens);

extern "C" enkf_fs_type *enkf_fs_create_fs(const char *mount_point,
                                           fs_driver_impl driver_id,
                                           unsigned ensemble_size, bool mount);

extern "C" void enkf_fs_umount(enkf_fs_type *fs);

char *enkf_fs_alloc_case_filename(const enkf_fs_type *fs,
                                  const char *input_name);
char *enkf_fs_alloc_case_tstep_filename(const enkf_fs_type *fs, int tstep,
                                        const char *input_name);
char *enkf_fs_alloc_case_tstep_member_filename(const enkf_fs_type *fs,
                                               int tstep, int iens,
                                               const char *input_name);

FILE *enkf_fs_open_case_tstep_file(const enkf_fs_type *fs,
                                   const char *input_name, int tstep,
                                   const char *mode);

FILE *enkf_fs_open_excase_file(const enkf_fs_type *fs, const char *input_name);
FILE *enkf_fs_open_excase_tstep_file(const enkf_fs_type *fs,
                                     const char *input_name, int tstep);

TimeMap &enkf_fs_get_time_map(const enkf_fs_type *fs);

#endif
