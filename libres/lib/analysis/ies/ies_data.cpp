#include <algorithm>
#include <memory>

#include <ert/analysis/ies/ies_data.hpp>

/*
  The configuration data used by the ies_enkf module is contained in a
  ies_data_struct instance. The data type used for the ies_enkf
  module is quite simple; with only a few scalar variables, but there
  are essentially no limits to what you can pack into such a datatype.

  All the functions in the module have a void pointer as the first argument,
  this will immediately be cast to an ies::data_type instance, to get some
  type safety the UTIL_TYPE_ID system should be used.

  The data structure holding the data for your analysis module should
  be created and initialized by a constructor, which should be
  registered with the '.alloc' element of the analysis table; in the
  same manner the desctruction of this data should be handled by a
  destructor or free() function registered with the .freef field of
  the analysis table.
*/

ies::data::Data::Data(int ens_size)
    : m_ens_size(ens_size), m_converged(false), m_iteration_nr(0),
      W(matrix_alloc(ens_size, ens_size)) {}

ies::data::Data::~Data() {
    matrix_free(this->W);

    if (this->m_ens_mask)
        bool_vector_free(this->m_ens_mask);
    if (this->m_obs_mask)
        bool_vector_free(this->m_obs_mask);
    if (this->m_obs_mask0)
        bool_vector_free(this->m_obs_mask0);
    if (this->A0)
        matrix_free(this->A0);
    if (this->E)
        matrix_free(this->E);
}

void ies::data::Data::iteration_nr(int iteration_nr) {
    this->m_iteration_nr = iteration_nr;
}

int ies::data::Data::iteration_nr() const { return this->m_iteration_nr; }

int ies::data::Data::inc_iteration_nr() { return ++this->m_iteration_nr; }

void ies::data::Data::update_ens_mask(const bool_vector_type *mask) {
    if (this->m_ens_mask)
        bool_vector_free(this->m_ens_mask);

    this->m_ens_mask = bool_vector_alloc_copy(mask);
}

int ies::data::Data::ens_size() const { return this->m_ens_size; }

void ies::data::Data::store_initial_obs_mask(const bool_vector_type *mask) {
    if (!this->m_obs_mask0)
        this->m_obs_mask0 = bool_vector_alloc_copy(mask);
}

void ies::data::Data::update_obs_mask(const bool_vector_type *mask) {
    if (this->m_obs_mask)
        bool_vector_free(this->m_obs_mask);

    this->m_obs_mask = bool_vector_alloc_copy(mask);
}

int ies::data::Data::obs_mask_size() const {
    return bool_vector_size(this->m_obs_mask);
}

int ies::data::Data::active_obs_count() const {
    return bool_vector_count_equal(this->m_obs_mask, true);
}

int ies::data::Data::ens_mask_size() const {
    return bool_vector_size(this->m_ens_mask);
}

void ies::data::Data::update_state_size(int state_size) {
    this->m_state_size = state_size;
}

/* We store the initial observation perturbations in E, corresponding to active data->obs_mask0
   in data->E. The unused rows in data->E corresponds to false data->obs_mask0 */
void ies::data::Data::store_initialE(const matrix_type *E0) {
    if (!this->E) {
        int obs_size_msk = this->obs_mask_size();
        int ens_size_msk = this->ens_mask_size();
        this->E = matrix_alloc(obs_size_msk, ens_size_msk);
        matrix_set(this->E, -999.9);

        int m = 0;
        for (int iobs = 0; iobs < obs_size_msk; iobs++) {
            if (bool_vector_iget(this->m_obs_mask0, iobs)) {
                int active_idx = 0;
                for (int iens = 0; iens < ens_size_msk; iens++) {
                    if (bool_vector_iget(this->m_ens_mask, iens)) {
                        matrix_iset_safe(this->E, iobs, iens,
                                         matrix_iget(E0, m, active_idx));
                        active_idx++;
                    }
                }
                m++;
            }
        }
    }
}

/* We augment the additional observation perturbations arriving in later iterations, that was not stored before,
   in data->E. */
void ies::data::Data::augment_initialE(const matrix_type *E0) {
    if (this->E) {
        int obs_size_msk = this->obs_mask_size();
        int ens_size_msk = this->ens_mask_size();
        int m = 0;
        for (int iobs = 0; iobs < obs_size_msk; iobs++) {
            if (!bool_vector_iget(this->m_obs_mask0, iobs) &&
                bool_vector_iget(this->m_obs_mask, iobs)) {
                int i = -1;
                for (int iens = 0; iens < ens_size_msk; iens++) {
                    if (bool_vector_iget(this->m_ens_mask, iens)) {
                        i++;
                        matrix_iset_safe(this->E, iobs, iens,
                                         matrix_iget(E0, m, i));
                    }
                }
                bool_vector_iset(this->m_obs_mask0, iobs, true);
            }
            if (bool_vector_iget(this->m_obs_mask, iobs)) {
                m++;
            }
        }
    }
}

void ies::data::Data::store_initialA(const matrix_type *A) {
    if (!this->A0)
        this->A0 = matrix_alloc_copy(A);
}

const bool_vector_type *ies::data::Data::obs_mask0() const {
    return this->m_obs_mask0;
}

const bool_vector_type *ies::data::Data::obs_mask() const {
    return this->m_obs_mask;
}

const bool_vector_type *ies::data::Data::ens_mask() const {
    return this->m_ens_mask;
}

const matrix_type *ies::data::Data::getE() const { return this->E; }

matrix_type *ies::data::Data::getW() { return this->W; }

const matrix_type *ies::data::Data::getW() const { return this->W; }

const matrix_type *ies::data::Data::getA0() const { return this->A0; }

namespace {

matrix_type *alloc_active(const matrix_type *full_matrix,
                          const bool_vector_type *row_mask,
                          const bool_vector_type *column_mask) {
    int rows = bool_vector_size(row_mask);
    int columns = bool_vector_size(column_mask);

    matrix_type *active =
        matrix_alloc(bool_vector_count_equal(row_mask, true),
                     bool_vector_count_equal(column_mask, true));
    int row = 0;
    for (int iobs = 0; iobs < rows; iobs++) {
        if (bool_vector_iget(row_mask, iobs)) {
            int column = 0;
            for (int iens = 0; iens < columns; iens++) {
                if (bool_vector_iget(column_mask, iens)) {
                    matrix_iset(active, row, column,
                                matrix_iget(full_matrix, iobs, iens));
                    column++;
                }
            }
            row++;
        }
    }

    return active;
}
} // namespace

/*
  During the iteration process both the number of realizations and the number of
  observations can change, the number of realizations can only be reduced but
  the number of (active) observations can both be reduced and increased. The
  iteration algorithm is based maintaining a state for the entire update
  process, in order to do this correctly we must create matrix representations
  with the correct active elements both in observation and realisation space.
*/

matrix_type *ies::data::Data::alloc_activeE() const {
    return alloc_active(this->E, this->m_obs_mask, this->m_ens_mask);
}

matrix_type *ies::data::Data::alloc_activeW() const {
    return alloc_active(this->W, this->m_ens_mask, this->m_ens_mask);
}

matrix_type *ies::data::Data::alloc_activeA() const {
    bool_vector_type *state_mask =
        bool_vector_alloc(matrix_get_rows(this->A0), true);
    auto *activeA = alloc_active(this->A0, state_mask, this->m_ens_mask);
    bool_vector_free(state_mask);
    return activeA;
}
