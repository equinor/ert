#ifndef IES_ENKF_DATA_H
#define IES_ENKF_DATA_H

#include <ert/util/rng.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/util/bool_vector.hpp>
#include <ert/util/type_macros.hpp>

#include <ert/analysis/ies/ies_enkf_config.hpp>

namespace ies {

typedef struct enkf_data_struct enkf_data_type;

void *enkf_data_alloc();
void enkf_data_free(void *arg);

void enkf_data_set_iteration_nr(enkf_data_type *data, int iteration_nr);
int enkf_data_get_iteration_nr(const enkf_data_type *data);
int enkf_data_inc_iteration_nr(enkf_data_type *data);

enkf_config_type *enkf_data_get_config(const enkf_data_type *data);

void enkf_data_update_ens_mask(enkf_data_type *data,
                               const bool_vector_type *ens_mask);
void enkf_store_initial_obs_mask(enkf_data_type *data,
                                 const bool_vector_type *obs_mask);
void enkf_update_obs_mask(enkf_data_type *data,
                          const bool_vector_type *obs_mask);
void enkf_data_update_state_size(enkf_data_type *data, int state_size);

int enkf_data_get_obs_mask_size(const enkf_data_type *data);
int enkf_data_get_ens_mask_size(const enkf_data_type *data);
int enkf_data_active_obs_count(const enkf_data_type *data);

const bool_vector_type *enkf_data_get_obs_mask0(const enkf_data_type *data);
const bool_vector_type *enkf_data_get_obs_mask(const enkf_data_type *data);
const bool_vector_type *enkf_data_get_ens_mask(const enkf_data_type *data);

FILE *enkf_data_open_log(enkf_data_type *data);
void enkf_data_fclose_log(enkf_data_type *data);

void enkf_data_allocateW(enkf_data_type *data);
void enkf_data_store_initialE(enkf_data_type *data, const matrix_type *E0);
void enkf_data_augment_initialE(enkf_data_type *data, const matrix_type *E0);
void enkf_data_store_initialA(enkf_data_type *data, const matrix_type *A);
const matrix_type *enkf_data_getE(const enkf_data_type *data);
const matrix_type *enkf_data_getA0(const enkf_data_type *data);
matrix_type *enkf_data_getW(const enkf_data_type *data);

UTIL_SAFE_CAST_HEADER(enkf_data);
UTIL_SAFE_CAST_HEADER_CONST(enkf_data);

} // namespace ies

#endif
