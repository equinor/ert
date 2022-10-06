#pragma once

#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/enkf_analysis.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/row_scaling.hpp>
#include <ert/util/rng.hpp>
#include <fmt/format.h>
#include <iterator>
#include <optional>
#include <stdexcept>

namespace analysis {
/**
 * Container for all data required for performing an update step.
 * Data consists of 5 matrices and a list of pairs of rowscaling and matrix.
 * objects mask describing the observations which
 * are active. In addition a flag has_observations which is used to determine wheter
 * it is possible to do an update step.
*/
class ObservationHandler
    : public std::enable_shared_from_this<ObservationHandler> {
public:
    ObservationHandler() = default;
    ObservationHandler(Eigen::VectorXd observation_values_in,
                       Eigen::VectorXd observation_errors_in,
                       const std::vector<bool> &obs_mask_in,
                       const UpdateSnapshot &update_snapshot_in)
        : observation_values(observation_values_in),
          observation_errors(observation_errors_in),
          obs_mask(std::move(obs_mask_in)),
          update_snapshot(std::move(update_snapshot_in)) {}

    Eigen::VectorXd observation_values;
    Eigen::VectorXd observation_errors;
    std::vector<bool> obs_mask;
    UpdateSnapshot update_snapshot;
};

class Parameter : public std::enable_shared_from_this<Parameter> {
public:
    std::string name;
    ActiveList active_list;

    Parameter(std::string name, const ActiveList &active_list = std::nullopt)
        : name(name), active_list(active_list) {}

    std::string to_string() const {
        if (active_list.has_value()) {
            return fmt::format("Parameter(name='{}', index_list=[{}])", name,
                               fmt::join(*active_list, ", "));
        } else {
            return fmt::format("Parameter(name='{}', index_list=[])", name);
        }
    }
};

class RowScalingParameter
    : public std::enable_shared_from_this<RowScalingParameter>,
      public Parameter {
public:
    std::shared_ptr<RowScaling> row_scaling;

    RowScalingParameter(std::string name,
                        std::shared_ptr<RowScaling> row_scaling,
                        const ActiveList &active_list = std::nullopt)
        : Parameter(name, active_list), row_scaling(std::move(row_scaling)) {}
};

} // namespace analysis
