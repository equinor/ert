#include <cmath>
#include <vector>

#include <ert/util/util.h>

#include <ert/analysis/analysis_module.hpp>
#include <ert/enkf/enkf_analysis.hpp>
#include <ert/enkf/meas_data.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/except.hpp>
#include <ert/python.hpp>

void UpdateSnapshot::add_member(std::string observation_name,
                                double observation_value,
                                double observation_error,
                                std::string observation_status,
                                double ensemble_mean, double ensemble_std) {
    obs_name_.push_back(observation_name);
    obs_value_.push_back(observation_value);
    obs_error_.push_back(observation_error);
    obs_status_.push_back(observation_status);
    response_mean_.push_back(ensemble_mean);
    response_std_.push_back(ensemble_std);
}

UpdateSnapshot make_update_snapshot(const obs_data_type *obs_data,
                                    const meas_data_type *meas_data) {
    UpdateSnapshot update_snapshot;

    for (int block_nr = 0; block_nr < obs_data_get_num_blocks(obs_data);
         block_nr++) {
        const obs_block_type *obs_block =
            obs_data_iget_block_const(obs_data, block_nr);
        meas_block_type *meas_block = meas_data_iget_block(meas_data, block_nr);
        const char *obs_key = obs_block_get_key(obs_block);
        for (int iobs = 0; iobs < obs_block_get_size(obs_block); iobs++) {
            active_type active_mode =
                obs_block_iget_active_mode(obs_block, iobs);
            std::string obs_status;
            if (active_mode == ACTIVE) {
                obs_status = "ACTIVE";
            } else if (active_mode == DEACTIVATED) {
                obs_status = "DEACTIVATED";
            } else if (active_mode == LOCAL_INACTIVE) {
                obs_status = "LOCAL_INACTIVE";
            } else if (active_mode == MISSING) {
                obs_status = "MISSING";
            } else
                util_abort("%s: enum_value:%d not handled - internal error\n",
                           __func__, active_mode);

            double response_mean;
            double response_std;
            if ((active_mode == MISSING) || (active_mode == LOCAL_INACTIVE)) {
                response_mean = NAN;
                response_std = NAN;
            } else {
                response_mean = meas_block_iget_ens_mean(meas_block, iobs);
                response_std = meas_block_iget_ens_std(meas_block, iobs);
            }
            update_snapshot.add_member(obs_key,
                                       obs_block_iget_value(obs_block, iobs),
                                       obs_block_iget_std(obs_block, iobs),
                                       obs_status, response_mean, response_std);
        }
    }
    return update_snapshot;
}

void enkf_analysis_deactivate_outliers(
    obs_data_type *obs_data, meas_data_type *meas_data, double std_cutoff,
    double alpha,
    const std::vector<std::pair<std::string, std::vector<int>>> &selected_obs) {
    for (int block_nr = 0; block_nr < obs_data_get_num_blocks(obs_data);
         block_nr++) {
        obs_block_type *obs_block = obs_data_iget_block(obs_data, block_nr);
        meas_block_type *meas_block = meas_data_iget_block(meas_data, block_nr);

        const std::vector<int> deactivate_index =
            selected_obs.at(block_nr).second;
        if (obs_block_get_key(obs_block) != selected_obs.at(block_nr).first)
            throw exc::invalid_argument("Expected obs_key: {}, got: {}",
                                        obs_block_get_key(obs_block),
                                        selected_obs.at(block_nr).first);

        int iobs;
        for (iobs = 0; iobs < meas_block_get_total_obs_size(meas_block);
             iobs++) {
            if (!deactivate_index.empty() &&
                std::find(deactivate_index.begin(), deactivate_index.end(),
                          iobs) == deactivate_index.end()) {
                obs_block_deactivate(obs_block, iobs,
                                     "User defined deactivation");
                meas_block_deactivate(meas_block, iobs);
                continue;
            }

            if (meas_block_iget_active(meas_block, iobs)) {
                double ens_std = meas_block_iget_ens_std(meas_block, iobs);
                if (ens_std <= std_cutoff) {
                    /*
                         * Deactivated because the ensemble has too small
                         * variation for this particular measurement.
                         */
                    obs_block_deactivate(obs_block, iobs,
                                         "No ensemble variation");
                    meas_block_deactivate(meas_block, iobs);
                } else {
                    double ens_mean =
                        meas_block_iget_ens_mean(meas_block, iobs);
                    double obs_std = obs_block_iget_std(obs_block, iobs);
                    double obs_value = obs_block_iget_value(obs_block, iobs);
                    double innov = obs_value - ens_mean;

                    /*
                         * Deactivated because the distance between the observed data
                         * and the ensemble prediction is to large. Keeping these
                         * outliers will lead to numerical problems.
                         */

                    if (std::abs(innov) > alpha * (ens_std + obs_std)) {
                        obs_block_deactivate(obs_block, iobs, "No overlap");
                        meas_block_deactivate(meas_block, iobs);
                    }
                }
            }
        }
    }
}

ERT_CLIB_SUBMODULE("enkf_analysis", m) {
    using namespace py::literals;
    py::class_<UpdateSnapshot>(m, "UpdateSnapshot")
        .def(py::init<>())
        .def_property_readonly("obs_name", &UpdateSnapshot::obs_name)
        .def_property_readonly("obs_value", &UpdateSnapshot::obs_value)
        .def_property_readonly("obs_std", &UpdateSnapshot::obs_error)
        .def_property_readonly("obs_status", &UpdateSnapshot::obs_status)
        .def_property_readonly("response_mean", &UpdateSnapshot::response_mean)
        .def_property_readonly("response_std", &UpdateSnapshot::response_std);
}
