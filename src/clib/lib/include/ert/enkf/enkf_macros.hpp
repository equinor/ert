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

#define VOID_UPDATE_STD_SCALE(prefix)                                          \
    void prefix##_update_std_scale__(void *void_obs, double std_multiplier,    \
                                     const ActiveList *active_list) {          \
        auto obs = static_cast<prefix##_type *>(void_obs);                     \
        prefix##_update_std_scale(obs, std_multiplier, active_list);           \
    }

#define VOID_UPDATE_STD_SCALE_HEADER(prefix)                                   \
    void prefix##_update_std_scale__(void *void_obs, double std_multiplier,    \
                                     const ActiveList *active_list);

#endif
