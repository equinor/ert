/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_node.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_ENKF_NODE_H
#define ERT_ENKF_NODE_H
#include <Eigen/Dense>
#include <stdbool.h>
#include <stdlib.h>

#include <ert/util/buffer.h>
#include <ert/util/hash.h>
#include <ert/util/int_vector.h>
#include <ert/util/rng.h>
#include <ert/util/type_macros.h>

#include <ert/ecl/ecl_file.h>
#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_sum.h>
#include <ert/ecl/fortio.h>

#include <ert/enkf/active_list.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_serialize.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/value_export.hpp>

typedef void(serialize_ftype)(const void *, node_id_type, const ActiveList *,
                              Eigen::MatrixXd &, int, int);
typedef void(deserialize_ftype)(void *, node_id_type, const ActiveList *,
                                const Eigen::MatrixXd &, int, int);

typedef void(ecl_write_ftype)(const void *, /* Node object */
                              const char *, /* Directory to write to. */
                              const char *, /* Filename - can be NULL. */
                              value_export_type *);

typedef bool(fload_ftype)(void *, const char *);
typedef void(read_from_buffer_ftype)(void *, buffer_type *, enkf_fs_type *,
                                     int);
typedef bool(write_to_buffer_ftype)(const void *, buffer_type *, int);
typedef bool(has_data_ftype)(const void *, int);

typedef void(user_get_vector_ftype)(void *, const char *, double_vector_type *);
typedef bool(user_get_ftype)(void *, const char *, int, double *);
typedef void *(alloc_ftype)(const void *);
typedef bool(initialize_ftype)(void *, int, const char *, rng_type *);
typedef bool(forward_load_ftype)(void *, const char *, int, const void *);
typedef bool(forward_load_vector_ftype)(void *, const char *,
                                        const ecl_sum_type *,
                                        const int_vector_type *);
typedef void(free_data_ftype)(void *);
typedef void(node_free_ftype)(void *);
typedef void(clear_ftype)(void *);
typedef void(node_copy_ftype)(const void *, void *);
typedef void(ensemble_mulX_vector_ftype)(void *, int, const void **,
                                         const double *);
void enkf_node_serialize(enkf_node_type *enkf_node, enkf_fs_type *fs,
                         node_id_type node_id, const ActiveList *active_list,
                         Eigen::MatrixXd &A, int row_offset, int column);
void enkf_node_deserialize(enkf_node_type *enkf_node, enkf_fs_type *fs,
                           node_id_type node_id, const ActiveList *active_list,
                           const Eigen::MatrixXd &A, int row_offset,
                           int column);

typedef enum {
    alloc_func = 0,
    ecl_write_func = 1,
    forward_load_func = 2,
    fread_func = 3,
    fwrite_func = 4,
    copy_func = 5,
    initialize_func = 6,
    free_func = 7,
    free_data_func = 8,
    clear_serial_state_func = 9,
    serialize = 10,
    deserialize = 11
} node_function_type;

typedef void(enkf_node_ftype1)(enkf_node_type *);

bool enkf_node_user_get_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                               const char *key, int iens,
                               double_vector_type *values);
bool enkf_node_user_get(enkf_node_type *, enkf_fs_type *, const char *,
                        node_id_type, double *);
enkf_node_type *enkf_node_deep_alloc(const enkf_config_node_type *config);
extern "C" enkf_node_type *enkf_node_alloc(const enkf_config_node_type *);
/*
    The enkf_node_free() function declaration is in the enkf_config_node.h header,
    because the enkf_config_node needs to know how to free the min_std node.

    void             enkf_node_free(enkf_node_type *enkf_node);
  */

bool enkf_node_forward_init(enkf_node_type *enkf_node, const char *run_path,
                            int iens);
extern "C" bool enkf_node_has_data(enkf_node_type *enkf_node, enkf_fs_type *fs,
                                   node_id_type node_id);
extern "C" void *enkf_node_value_ptr(const enkf_node_type *);
extern "C" ert_impl_type enkf_node_get_impl_type(const enkf_node_type *);
bool enkf_node_use_forward_init(const enkf_node_type *enkf_node);

bool enkf_node_forward_load_vector(enkf_node_type *enkf_node,
                                   const ecl_sum_type *ecl_sum,
                                   const int_vector_type *time_index);
bool enkf_node_forward_load(enkf_node_type *enkf_node, int report_step,
                            const run_arg_type *run_arg,
                            const ecl_sum_type *ecl_sum);
void enkf_node_ecl_write(const enkf_node_type *, const char *,
                         value_export_type *, int);
bool enkf_node_initialize(enkf_node_type *enkf_node, int, rng_type *);

void enkf_node_copy(const enkf_config_node_type *config_node,
                    enkf_fs_type *src_case, enkf_fs_type *target_case,
                    node_id_type src_id, node_id_type target_id);
enkf_node_type *enkf_node_load_alloc(const enkf_config_node_type *config_node,
                                     enkf_fs_type *fs, node_id_type node_id);
bool enkf_node_fload(enkf_node_type *enkf_node, const char *filename);
void enkf_node_load(enkf_node_type *enkf_node, enkf_fs_type *fs,
                    node_id_type node_id);
void enkf_node_load_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                           int iens);
extern "C" bool enkf_node_store(enkf_node_type *enkf_node, enkf_fs_type *fs,
                                node_id_type node_id);
bool enkf_node_store_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                            int iens);
extern "C" bool enkf_node_try_load(enkf_node_type *enkf_node, enkf_fs_type *fs,
                                   node_id_type node_id);
bool enkf_node_try_load_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                               int iens);
bool enkf_node_vector_storage(const enkf_node_type *node);

const enkf_config_node_type *enkf_node_get_config(const enkf_node_type *);
extern "C" const char *enkf_node_get_key(const enkf_node_type *);

UTIL_IS_INSTANCE_HEADER(enkf_node);

#endif
