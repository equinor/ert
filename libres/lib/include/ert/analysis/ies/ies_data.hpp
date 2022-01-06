#ifndef IES_DATA_H
#define IES_DATA_H

#include <ert/util/rng.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/util/bool_vector.hpp>
#include <ert/util/type_macros.hpp>

#include <ert/analysis/ies/ies_config.hpp>

namespace ies {

typedef struct data_struct data_type;

void *data_alloc();
void data_free(void *arg);

void data_set_iteration_nr(data_type *data, int iteration_nr);
int data_get_iteration_nr(const data_type *data);
int data_inc_iteration_nr(data_type *data);

config::config_type *data_get_config(const data_type *data);

void data_update_ens_mask(data_type *data, const bool_vector_type *ens_mask);
void store_initial_obs_mask(data_type *data, const bool_vector_type *obs_mask);
void update_obs_mask(data_type *data, const bool_vector_type *obs_mask);
void data_update_state_size(data_type *data, int state_size);

int data_get_obs_mask_size(const data_type *data);
int data_get_ens_mask_size(const data_type *data);
int data_active_obs_count(const data_type *data);

const bool_vector_type *data_get_obs_mask0(const data_type *data);
const bool_vector_type *data_get_obs_mask(const data_type *data);
const bool_vector_type *data_get_ens_mask(const data_type *data);

FILE *data_open_log(data_type *data);
void data_fclose_log(data_type *data);

void data_allocateW(data_type *data);
void data_store_initialE(data_type *data, const matrix_type *E0);
void data_augment_initialE(data_type *data, const matrix_type *E0);
void data_store_initialA(data_type *data, const matrix_type *A);
const matrix_type *data_getE(const data_type *data);
const matrix_type *data_getA0(const data_type *data);
matrix_type *data_getW(const data_type *data);

UTIL_SAFE_CAST_HEADER(data);
UTIL_SAFE_CAST_HEADER_CONST(data);

} // namespace ies

#endif
