#ifndef ERT_ENKF_MACROS_H
#define ERT_ENKF_MACROS_H

#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>

#include <ert/util/double_vector.hpp>
#include <ert/util/int_vector.hpp>

#include <ert/ecl/ecl_file.hpp>
#include <ert/ecl/ecl_sum.hpp>

#include <ert/enkf/active_list.hpp>
#include <ert/enkf/enkf_fs_type.hpp>
#include <ert/enkf/enkf_types.hpp>

#define VOID_CONFIG_FREE(prefix)                                               \
    void prefix##_config_free__(void *void_arg) {                              \
        prefix##_config_free((prefix##_config_type *)void_arg);                \
    }
#define VOID_CONFIG_FREE_HEADER(prefix) void prefix##_config_free__(void *);

#define GET_DATA_SIZE(prefix)                                                  \
    int prefix##_config_get_data_size(const prefix##_config_type *arg) {       \
        return arg->data_size;                                                 \
    }
#define GET_DATA_SIZE_HEADER(prefix)                                           \
    int prefix##_config_get_data_size(const prefix##_config_type *arg);

#define VOID_GET_DATA_SIZE(prefix)                                             \
    int prefix##_config_get_data_size__(const void *arg) {                     \
        auto config = static_cast<const prefix##_config_type *>(arg);          \
        return prefix##_config_get_data_size(config);                          \
    }
#define VOID_GET_DATA_SIZE_HEADER(prefix)                                      \
    int prefix##_config_get_data_size__(const void *arg);

#define VOID_ALLOC(prefix)                                                     \
    void *prefix##_alloc__(const void *void_config) {                          \
        auto config = static_cast<const prefix##_config_type *>(void_config);  \
        return prefix##_alloc(config);                                         \
    }

#define VOID_ALLOC_HEADER(prefix) void *prefix##_alloc__(const void *);

#define VOID_HAS_DATA(prefix)                                                  \
    bool prefix##_has_data__(const void *void_arg, int report_step) {          \
        auto arg = static_cast<const prefix##_type *>(void_arg);               \
        return prefix##_has_data(arg, report_step);                            \
    }

#define VOID_HAS_DATA_HEADER(prefix)                                           \
    bool prefix##_has_data__(const void *, int);

#define VOID_WRITE_TO_BUFFER(prefix)                                           \
    bool prefix##_write_to_buffer__(const void *void_arg, buffer_type *buffer, \
                                    int report_step) {                         \
        auto arg = static_cast<const prefix##_type *>(void_arg);               \
        return prefix##_write_to_buffer(arg, buffer, report_step);             \
    }

#define VOID_READ_FROM_BUFFER(prefix)                                          \
    void prefix##_read_from_buffer__(void *void_arg, buffer_type *buffer,      \
                                     enkf_fs_type *fs, int report_step) {      \
        auto arg = static_cast<prefix##_type *>(void_arg);                     \
        prefix##_read_from_buffer(arg, buffer, fs, report_step);               \
    }

#define VOID_WRITE_TO_BUFFER_HEADER(prefix)                                    \
    bool prefix##_write_to_buffer__(const void *, buffer_type *, int);
#define VOID_READ_FROM_BUFFER_HEADER(prefix)                                   \
    void prefix##_read_from_buffer__(void *, buffer_type *, enkf_fs_type *,    \
                                     int);

#define VOID_FREE(prefix)                                                      \
    void prefix##_free__(void *void_arg) {                                     \
        auto arg = static_cast<prefix##_type *>(void_arg);                     \
        prefix##_free(arg);                                                    \
    }

#define VOID_FREE_HEADER(prefix) void prefix##_free__(void *);

#define VOID_USER_GET_OBS(prefix)                                              \
    void prefix##_user_get__(void *void_arg, const char *key, double *value,   \
                             double *std, bool *valid) {                       \
        auto arg = static_cast<prefix##_type *>(void_arg);                     \
        prefix##_user_get(arg, key, value, std, valid);                        \
    }

#define VOID_USER_GET_OBS_HEADER(prefix)                                       \
    void prefix##_user_get__(void *, const char *, double *, double *, bool *);

#define CONFIG_GET_ECL_KW_NAME(prefix)                                         \
    const char *prefix##_config_get_ecl_kw_name(                               \
        const prefix##_config_type *config) {                                  \
        return config->ecl_kw_name;                                            \
    }
#define CONFIG_GET_ECL_KW_NAME_HEADER(prefix)                                  \
    const char *prefix##_config_get_ecl_kw_name(const prefix##_config_type *)

#define VOID_SERIALIZE(prefix)                                                 \
    void prefix##_serialize__(const void *void_arg, node_id_type node_id,      \
                              const ActiveList *active_list,                   \
                              Eigen::MatrixXd &A, int row_offset,              \
                              int column) {                                    \
        auto arg = static_cast<const prefix##_type *>(void_arg);               \
        prefix##_serialize(arg, node_id, active_list, A, row_offset, column);  \
    }
#define VOID_SERIALIZE_HEADER(prefix)                                          \
    void prefix##_serialize__(const void *, node_id_type, const ActiveList *,  \
                              Eigen::MatrixXd &, int, int);

#define VOID_DESERIALIZE(prefix)                                               \
    void prefix##_deserialize__(                                               \
        void *void_arg, node_id_type node_id, const ActiveList *active_list,   \
        const Eigen::MatrixXd &A, int row_offset, int column) {                \
        auto arg = static_cast<prefix##_type *>(void_arg);                     \
        prefix##_deserialize(arg, node_id, active_list, A, row_offset,         \
                             column);                                          \
    }
#define VOID_DESERIALIZE_HEADER(prefix)                                        \
    void prefix##_deserialize__(void *, node_id_type, const ActiveList *,      \
                                const Eigen::MatrixXd &, int, int);

#define VOID_INITIALIZE(prefix)                                                \
    bool prefix##_initialize__(void *void_arg, int iens,                       \
                               const char *init_file) {                        \
        auto arg = static_cast<prefix##_type *>(void_arg);                     \
        return prefix##_initialize(arg, iens, init_file);                      \
    }
#define VOID_INITIALIZE_HEADER(prefix)                                         \
    bool prefix##_initialize__(void *, int, const char *);

#define VOID_UPDATE_STD_SCALE(prefix)                                          \
    void prefix##_update_std_scale__(void *void_obs, double std_multiplier,    \
                                     const ActiveList *active_list) {          \
        auto obs = static_cast<prefix##_type *>(void_obs);                     \
        prefix##_update_std_scale(obs, std_multiplier, active_list);           \
    }

#define VOID_UPDATE_STD_SCALE_HEADER(prefix)                                   \
    void prefix##_update_std_scale__(void *void_obs, double std_multiplier,    \
                                     const ActiveList *active_list);

#define VOID_CLEAR(prefix)                                                     \
    void prefix##_clear__(void *void_arg) {                                    \
        prefix##_clear(static_cast<prefix##_type *>(void_arg));                \
    }
#endif
