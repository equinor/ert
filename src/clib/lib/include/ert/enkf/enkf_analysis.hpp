#ifndef ERT_ENKF_ANALYSIS_H
#define ERT_ENKF_ANALYSIS_H
#include <stdio.h>
#include <vector>

#include <ert/util/int_vector.h>

#include <ert/enkf/obs_data.hpp>

class UpdateSnapshot {

private:
    std::vector<std::string> obs_name_;
    std::vector<double> obs_value_;
    std::vector<double> obs_error_;
    std::vector<std::string> obs_status_;
    std::vector<double> response_mean_;
    std::vector<double> response_std_;

public:
    const std::vector<std::string> &obs_name() const { return obs_name_; }
    const std::vector<double> &obs_value() const { return obs_value_; }
    const std::vector<double> &obs_error() const { return obs_error_; }
    const std::vector<std::string> &obs_status() const { return obs_status_; }
    const std::vector<double> &response_mean() const { return response_mean_; }
    const std::vector<double> &response_std() const { return response_std_; }

    void add_member(std::string observation_name, double observation_value,
                    double observation_error, std::string observation_status,
                    double ensemble_mean, double ensemble_std);
};

UpdateSnapshot make_update_snapshot(const obs_data_type *obs_data,
                                    const meas_data_type *meas_data);

void enkf_analysis_deactivate_outliers(
    obs_data_type *obs_data, meas_data_type *meas_data, double std_cutoff,
    double alpha,
    const std::vector<std::pair<std::string, std::vector<int>>> &selected_obs);

#endif
