#ifndef ERT_ENKF_NODE_H
#define ERT_ENKF_NODE_H
#include <Eigen/Dense>
#include <stdbool.h>
#include <stdlib.h>

#include <ert/util/buffer.h>
#include <ert/util/hash.h>
#include <ert/util/int_vector.h>

#include <ert/ecl/ecl_file.h>
#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_sum.h>
#include <ert/ecl/fortio.h>

#include <ert/enkf/active_list.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/enkf_util.hpp>

typedef void(serialize_ftype)(const void *, node_id_type, const ActiveList *,
                              Eigen::MatrixXd &, int, int);
typedef void(deserialize_ftype)(void *, node_id_type, const ActiveList *,
                                const Eigen::MatrixXd &, int, int);

typedef bool(fload_ftype)(void *, const char *);
typedef void(read_from_buffer_ftype)(void *, buffer_type *, enkf_fs_type *,
                                     int);
typedef bool(write_to_buffer_ftype)(const void *, buffer_type *, int);
typedef bool(has_data_ftype)(const void *, int);

typedef void *(alloc_ftype)(const void *);
typedef bool(initialize_ftype)(void *, int, const char *);
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

typedef void(enkf_node_ftype1)(enkf_node_type *);

enkf_node_type *enkf_node_deep_alloc(const enkf_config_node_type *config);
extern "C" enkf_node_type *enkf_node_alloc(const enkf_config_node_type *);

bool enkf_node_forward_init(enkf_node_type *enkf_node,
                            const std::string &run_path, int iens);
extern "C" bool enkf_node_has_data(enkf_node_type *enkf_node, enkf_fs_type *fs,
                                   node_id_type node_id);
extern "C" void *enkf_node_value_ptr(const enkf_node_type *);
extern "C" ert_impl_type enkf_node_get_impl_type(const enkf_node_type *);

bool enkf_node_forward_load(enkf_node_type *enkf_node, int report_step,
                            const std::string &run_path, enkf_fs_type *fs);
bool enkf_node_initialize(enkf_node_type *enkf_node, int);

void enkf_node_copy(const enkf_config_node_type *config_node,
                    enkf_fs_type *src_case, enkf_fs_type *target_case,
                    node_id_type src_id, node_id_type target_id);
enkf_node_type *enkf_node_load_alloc(const enkf_config_node_type *config_node,
                                     enkf_fs_type *fs, node_id_type node_id);
void enkf_node_load(enkf_node_type *enkf_node, enkf_fs_type *fs,
                    node_id_type node_id);
void enkf_node_load_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                           int iens);
extern "C" bool enkf_node_store(enkf_node_type *enkf_node, enkf_fs_type *fs,
                                node_id_type node_id);
bool enkf_node_store_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                            int iens);
bool enkf_node_try_load(enkf_node_type *enkf_node, enkf_fs_type *fs,
                        node_id_type node_id);
bool enkf_node_try_load_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                               int iens);

const enkf_config_node_type *enkf_node_get_config(const enkf_node_type *);
extern "C" const char *enkf_node_get_key(const enkf_node_type *);

#endif
