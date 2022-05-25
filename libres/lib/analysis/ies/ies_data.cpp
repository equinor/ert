#include <algorithm>
#include <memory>

#include <ert/analysis/ies/ies_data.hpp>
#include <ert/python.hpp>

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
    : W(Eigen::MatrixXd::Zero(ens_size, ens_size)) {}

void ies::data::Data::update_ens_mask(const std::vector<bool> &mask) {
    this->m_ens_mask = mask;
}

void ies::data::Data::store_initial_obs_mask(const std::vector<bool> &mask) {
    if (this->m_obs_mask0.empty())
        this->m_obs_mask0 = mask;
}

void ies::data::Data::update_obs_mask(const std::vector<bool> &mask) {
    this->m_obs_mask = mask;
}

int ies::data::Data::obs_mask_size() const { return this->m_obs_mask.size(); }

int ies::data::Data::ens_mask_size() const { return (this->m_ens_mask.size()); }

void ies::data::Data::update_state_size(int state_size) {
    this->m_state_size = state_size;
}

/* We store the initial observation perturbations in E, corresponding to active data->obs_mask0
   in data->E. The unused rows in data->E corresponds to false data->obs_mask0 */
void ies::data::Data::store_initialE(const Eigen::MatrixXd &E0) {
    if (E.rows() != 0 || E.cols() != 0)
        return;
    int obs_size_msk = this->obs_mask_size();
    int ens_size_msk = this->ens_mask_size();
    this->E = Eigen::MatrixXd::Zero(obs_size_msk, ens_size_msk);
    this->E.setConstant(-999.9);

    int m = 0;
    for (int iobs = 0; iobs < obs_size_msk; iobs++) {
        if (this->m_obs_mask0[iobs]) {
            int active_idx = 0;
            for (int iens = 0; iens < ens_size_msk; iens++) {
                if (this->m_ens_mask[iens]) {
                    this->E(iobs, iens) = E0(m, active_idx);
                    active_idx++;
                }
            }
            m++;
        }
    }
}

/* We augment the additional observation perturbations arriving in later iterations, that was not stored before,
   in data->E. */
void ies::data::Data::augment_initialE(const Eigen::MatrixXd &E0) {

    int obs_size_msk = this->obs_mask_size();
    int ens_size_msk = this->ens_mask_size();
    int m = 0;
    for (int iobs = 0; iobs < obs_size_msk; iobs++) {
        if (!this->m_obs_mask0[iobs] && this->m_obs_mask[iobs]) {
            int i = -1;
            for (int iens = 0; iens < ens_size_msk; iens++) {
                if (this->m_ens_mask[iens]) {
                    i++;
                    this->E(iobs, iens) = E0(m, i);
                }
            }
            this->m_obs_mask0[iobs] = true;
        }
        if (this->m_obs_mask[iobs]) {
            m++;
        }
    }
}

void ies::data::Data::store_initialA(const Eigen::MatrixXd &A) {
    if (A0.rows() == 0 || A0.cols() == 0)
        this->A0 = A;
}

const std::vector<bool> &ies::data::Data::obs_mask0() const {
    return this->m_obs_mask0;
}

const std::vector<bool> &ies::data::Data::obs_mask() const {
    return this->m_obs_mask;
}

const std::vector<bool> &ies::data::Data::ens_mask() const {
    return this->m_ens_mask;
}

const Eigen::MatrixXd &ies::data::Data::getE() const { return this->E; }

Eigen::MatrixXd &ies::data::Data::getW() { return this->W; }

const Eigen::MatrixXd &ies::data::Data::getW() const { return this->W; }

const Eigen::MatrixXd &ies::data::Data::getA0() const { return this->A0; }

namespace {

Eigen::MatrixXd make_active(const Eigen::MatrixXd &full_matrix,
                            const std::vector<bool> &row_mask,
                            const std::vector<bool> &column_mask) {
    int rows = row_mask.size();
    int columns = column_mask.size();

    Eigen::MatrixXd active = Eigen::MatrixXd::Zero(
        std::count(row_mask.begin(), row_mask.end(), true),
        std::count(column_mask.begin(), column_mask.end(), true));
    int row = 0;
    for (int iobs = 0; iobs < rows; iobs++) {
        if (row_mask[iobs]) {
            int column = 0;
            for (int iens = 0; iens < columns; iens++) {
                if (column_mask[iens]) {
                    active(row, column) = full_matrix(iobs, iens);
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

Eigen::MatrixXd ies::data::Data::make_activeE() const {
    return make_active(this->E, this->m_obs_mask, this->m_ens_mask);
}

Eigen::MatrixXd ies::data::Data::make_activeW() const {
    return make_active(this->W, this->m_ens_mask, this->m_ens_mask);
}

Eigen::MatrixXd ies::data::Data::make_activeA() const {
    std::vector<bool> state_mask(this->A0.rows(), true);
    return make_active(this->A0, state_mask, this->m_ens_mask);
}

RES_LIB_SUBMODULE("ies", m) {
    py::class_<ies::data::Data, std::shared_ptr<ies::data::Data>>(m,
                                                                  "ModuleData")
        .def(py::init<int>())
        .def_readwrite("iteration_nr", &ies::data::Data::iteration_nr);
}
