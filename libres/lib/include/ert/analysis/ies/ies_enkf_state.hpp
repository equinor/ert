#ifndef IES_ENKF_STATE_H
#define IES_ENKF_STATE_H

#include <ert/util/rng.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/util/bool_vector.hpp>
#include <ert/util/type_macros.hpp>

#include <ert/analysis/ies/ies_enkf_config.hpp>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ies_enkf_state_struct ies_enkf_state_type;

void *ies_enkf_state_alloc();
void ies_enkf_state_free(void *arg);

void ies_enkf_state_set_iteration_nr(ies_enkf_state_type *data,
                                     int iteration_nr);
int ies_enkf_state_get_iteration_nr(const ies_enkf_state_type *data);
int ies_enkf_state_inc_iteration_nr(ies_enkf_state_type *data);

ies_enkf_config_type *
ies_enkf_state_get_config(const ies_enkf_state_type *data);

void ies_enkf_state_update_ens_mask(ies_enkf_state_type *data,
                                    const bool_vector_type *ens_mask);
void ies_enkf_store_initial_obs_mask(ies_enkf_state_type *data,
                                     const bool_vector_type *obs_mask);
void ies_enkf_update_obs_mask(ies_enkf_state_type *data,
                              const bool_vector_type *obs_mask);
void ies_enkf_state_update_state_size(ies_enkf_state_type *data,
                                      int state_size);

int ies_enkf_state_get_obs_mask_size(const ies_enkf_state_type *data);
int ies_enkf_state_get_ens_mask_size(const ies_enkf_state_type *data);
int ies_enkf_state_active_obs_count(const ies_enkf_state_type *data);

const bool_vector_type *
ies_enkf_state_get_obs_mask0(const ies_enkf_state_type *data);
const bool_vector_type *
ies_enkf_state_get_obs_mask(const ies_enkf_state_type *data);
const bool_vector_type *
ies_enkf_state_get_ens_mask(const ies_enkf_state_type *data);

FILE *ies_enkf_state_open_log(ies_enkf_state_type *data);
void ies_enkf_state_fclose_log(ies_enkf_state_type *data);

void ies_enkf_state_allocateW(ies_enkf_state_type *data, int ens_size);
void ies_enkf_state_store_initialE(ies_enkf_state_type *data,
                                   const matrix_type *E0);
void ies_enkf_state_augment_initialE(ies_enkf_state_type *data,
                                     const matrix_type *E0);
void ies_enkf_state_store_initialA(ies_enkf_state_type *data,
                                   const matrix_type *A);
const matrix_type *ies_enkf_state_getE(const ies_enkf_state_type *data);
const matrix_type *ies_enkf_state_getA0(const ies_enkf_state_type *data);
matrix_type *ies_enkf_state_getW(const ies_enkf_state_type *data);

UTIL_SAFE_CAST_HEADER(ies_enkf_state);
UTIL_SAFE_CAST_HEADER_CONST(ies_enkf_state);

#ifdef __cplusplus
}
#endif

#endif
