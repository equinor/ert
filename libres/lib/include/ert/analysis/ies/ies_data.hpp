#ifndef IES_DATA_H
#define IES_DATA_H

#include <ert/util/rng.hpp>
#include <ert/res_util/matrix.hpp>
#include <ert/util/bool_vector.hpp>

#include <ert/analysis/ies/ies_config.hpp>

namespace ies {
namespace data {

constexpr const char *ITER_KEY = "ITER";

typedef struct data_struct data_type;

data_type *alloc(int ens_size, bool ies_mode);
void free(data_type *data);
int ens_size(const data_type *data);

void set_iteration_nr(data_type *data, int iteration_nr);
int get_iteration_nr(const data_type *data);
int inc_iteration_nr(data_type *data);

::ies::config::Config &get_config(const data_type *data);

void update_ens_mask(data_type *data, const bool_vector_type *ens_mask);
void store_initial_obs_mask(data_type *data, const bool_vector_type *obs_mask);
void update_obs_mask(data_type *data, const bool_vector_type *obs_mask);
void update_state_size(data_type *data, int state_size);

int get_obs_mask_size(const data_type *data);
int get_ens_mask_size(const data_type *data);
int active_obs_count(const data_type *data);

const bool_vector_type *get_obs_mask0(const data_type *data);
const bool_vector_type *get_obs_mask(const data_type *data);
const bool_vector_type *get_ens_mask(const data_type *data);

void store_initialE(data_type *data, const matrix_type *E0);
void augment_initialE(data_type *data, const matrix_type *E0);
void store_initialA(data_type *data, const matrix_type *A);
const matrix_type *getE(const data_type *data);
const matrix_type *getA0(const data_type *data);
matrix_type *getW(const data_type *data);

} // namespace data
} // namespace ies

#endif
