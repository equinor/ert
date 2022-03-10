#pragma once

#include <functional>
#include <stdexcept>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/local_updatestep.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/util/rng.hpp>

namespace analysis {
/**
 * Container for all data required for performing an update step.
 * Data consists of 5 matrices and a list of pairs of rowscaling and matrix.
 * objects mask describing the observations which
 * are active. In addition a flag has_observations which is used to determine wheter
 * it is possible to do an update step.
*/
class update_data_type : public std::enable_shared_from_this<update_data_type> {
public:
    update_data_type() = default;
    update_data_type(
        matrix_type S_in, matrix_type E_in, matrix_type D_in, matrix_type R_in,
        std::optional<Eigen::MatrixXd> A_in,
        std::vector<std::pair<Eigen::MatrixXd,
                              std::reference_wrapper<const RowScaling>>>
            A_with_rowscaling_in,
        const std::vector<bool> &obs_mask_in)
        : S(std::move(S_in)), E(std::move(E_in)), D(std::move(D_in)),
          R(std::move(R_in)), A(std::move(A_in)),
          obs_mask(std::move(obs_mask_in)),
          A_with_rowscaling(std::move(A_with_rowscaling_in)) {
        has_observations = true;
    }

    Eigen::MatrixXd S;
    Eigen::MatrixXd E;
    Eigen::MatrixXd D;
    Eigen::MatrixXd R;
    std::optional<Eigen::MatrixXd> A;
    std::vector<bool> obs_mask;
    std::vector<
        std::pair<Eigen::MatrixXd, std::reference_wrapper<const RowScaling>>>
        A_with_rowscaling;
    bool has_observations = false;
};
} // namespace analysis
