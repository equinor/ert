#include "ert/enkf/ensemble_config.hpp"
#include <Eigen/Dense>
#include <cerrno>
#include <fmt/format.h>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/update.hpp>
#include <ert/enkf/enkf_analysis.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/meas_data.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/except.hpp>
#include <ert/python.hpp>
#include <ert/res_util/memory.hpp>
#include <ert/res_util/metric.hpp>

static auto logger = ert::get_logger("analysis.update");

namespace analysis {

namespace {
std::vector<int>
bool_vector_to_active_list(const std::vector<bool> &bool_vector) {
    std::vector<int> active_list;
    for (int i = 0; i < bool_vector.size(); i++) {
        if (bool_vector[i])
            active_list.push_back(i);
    }
    return active_list;
}
} // namespace

/**
 This is very awkward; the problem is that for the GEN_DATA type the config
 object does not really own the size. Instead the size is pushed (on load time)
 from gen_data instances to the gen_data_config instance. Therefore we have to
 assert that at least one gen_data instance has been loaded (and consequently
 updated the gen_data_config instance) before calling enkf_config_node_get_data_size.
*/
void ensure_node_loaded(const enkf_config_node_type *config_node,
                        enkf_fs_type *fs) {
    if (enkf_config_node_get_impl_type(config_node) == GEN_DATA) {
        enkf_node_type *node = enkf_node_alloc(config_node);
        node_id_type node_id = {.report_step = 0, .iens = 0};

        enkf_node_load(node, fs, node_id);

        enkf_node_free(node);
    }
}

void serialize_node(enkf_fs_type *fs, const enkf_config_node_type *config_node,
                    int iens, int row_offset, int column,
                    const ActiveList *active_list, Eigen::MatrixXd &A) {

    enkf_node_type *node = enkf_node_alloc(config_node);
    node_id_type node_id = {.report_step = 0, .iens = iens};
    enkf_node_serialize(node, fs, node_id, active_list, A, row_offset, column);
    enkf_node_free(node);
}

void serialize_parameter(const ensemble_config_type *ens_config,
                         const Parameter parameter, enkf_fs_type *target_fs,
                         const std::vector<int> &iens_active_index,
                         Eigen::MatrixXd &A) {

    int ens_size = A.cols();

    const enkf_config_node_type *config_node =
        ensemble_config_get_node(ens_config, parameter.name.c_str());

    ensure_node_loaded(config_node, target_fs);
    int active_size = parameter.active_list.active_size(
        enkf_config_node_get_data_size(config_node, 0));

    if (active_size > A.rows())
        A.conservativeResize(A.rows() + active_size, ens_size);
    if (active_size > 0) {
        for (int column = 0; column < ens_size; column++) {
            int iens = iens_active_index[column];
            serialize_node(target_fs, config_node, iens, 0, column,
                           &parameter.active_list, A);
        }
    }

    A.conservativeResize(active_size, ens_size);
}

void deserialize_node(enkf_fs_type *fs,
                      const enkf_config_node_type *config_node, int iens,
                      int row_offset, int column, const ActiveList *active_list,
                      const Eigen::MatrixXd &A) {

    node_id_type node_id = {.report_step = 0, .iens = iens};
    enkf_node_type *node = enkf_node_alloc(config_node);

    enkf_node_deserialize(node, fs, node_id, active_list, A, row_offset,
                          column);
    enkf_fs_get_state_map(fs).update_matching(iens, STATE_UNDEFINED,
                                              STATE_INITIALIZED);
    enkf_node_free(node);
}

void assert_matrix_size(const Eigen::MatrixXd &m, const char *name, int rows,
                        int columns) {
    if (!((m.rows() == rows) && (m.cols() == columns)))
        throw exc::invalid_argument(
            "matrix mismatch {}:[{},{}] - expected:[{},{}]", name, m.rows(),
            m.cols(), rows, columns);
}

/**
load a  parameters from a enkf_fs_type storage into a
matrix.
*/
Eigen::MatrixXd load_parameter(Cwrap<enkf_fs_type> source_fs,
                               Cwrap<ensemble_config_type> ensemble_config,
                               const std::vector<int> &iens_active_index,
                               const Parameter parameter) {

    int active_ens_size = iens_active_index.size();
    int matrix_start_size = 250000;
    Eigen::MatrixXd A =
        Eigen::MatrixXd::Zero(matrix_start_size, active_ens_size);

    serialize_parameter(ensemble_config, parameter, source_fs,
                        iens_active_index, A);
    return A;
}

/**
save a paramater matrix to enkf_fs_type storage
*/
void save_parameter(Cwrap<enkf_fs_type> target_fs,
                    Cwrap<ensemble_config_type> ensemble_config,
                    const std::vector<int> &iens_active_index,
                    const Parameter parameter, const Eigen::MatrixXd &A) {

    int ens_size = iens_active_index.size();
    const enkf_config_node_type *config_node =
        ensemble_config_get_node(ensemble_config, parameter.name.c_str());

    int active_size = parameter.active_list.active_size(
        enkf_config_node_get_data_size(config_node, 0));
    if (active_size > 0) {
        for (int column = 0; column < ens_size; column++) {
            int iens = iens_active_index[column];
            deserialize_node(target_fs, config_node, iens, 0, column,
                             &parameter.active_list, A);
        }
    }
}

std::pair<Eigen::MatrixXd, ObservationHandler> load_observations_and_responses(
    enkf_fs_type *source_fs, enkf_obs_type *obs, double alpha,
    double std_cutoff, double global_std_scaling,
    const std::vector<bool> &ens_mask,
    const std::vector<std::pair<std::string, std::vector<int>>>
        &selected_observations) {
    /*
    Observations and measurements are collected in these temporary
    structures. obs_data is a precursor for the 'd' vector, and
    meas_data is a precursor for the 'S' matrix'.

    The reason for going via these temporary structures is to support
    deactivating observations which should not be used in the update
    process.
    */

    obs_data_type *obs_data = obs_data_alloc(global_std_scaling);
    meas_data_type *meas_data = meas_data_alloc(ens_mask);

    std::vector<int> ens_active_list = bool_vector_to_active_list(ens_mask);
    enkf_obs_get_obs_and_measure_data(obs, source_fs, selected_observations,
                                      ens_active_list, meas_data, obs_data);
    enkf_analysis_deactivate_outliers(obs_data, meas_data, std_cutoff, alpha,
                                      selected_observations);
    auto update_snapshot = make_update_snapshot(obs_data, meas_data);

    int active_obs_size = obs_data_get_active_size(obs_data);
    int active_ens_size = meas_data_get_active_ens_size(meas_data);
    Eigen::MatrixXd S = meas_data_makeS(meas_data);
    assert_matrix_size(S, "S", active_obs_size, active_ens_size);
    meas_data_free(meas_data);

    Eigen::VectorXd observation_values = obs_data_values_as_vector(obs_data);
    // Inflating measurement errors by a factor sqrt(global_std_scaling) as shown
    // in for example evensen2018 - Analysis of iterative ensemble smoothers for solving inverse problems.
    // `global_std_scaling` is 1.0 for ES.
    Eigen::VectorXd observation_errors =
        obs_data_errors_as_vector(obs_data) * sqrt(global_std_scaling);
    std::vector<bool> obs_mask = obs_data_get_active_mask(obs_data);

    obs_data_free(obs_data);

    return std::pair<Eigen::MatrixXd, ObservationHandler>(
        S, ObservationHandler(observation_values, observation_errors, obs_mask,
                              update_snapshot));
}
} // namespace analysis

namespace {
static Eigen::MatrixXd generate_noise(int active_obs_size, int active_ens_size,
                                      Cwrap<rng_type> shared_rng) {
    Eigen::MatrixXd noise =
        Eigen::MatrixXd::Zero(active_obs_size, active_ens_size);
    for (int j = 0; j < active_ens_size; j++)
        for (int i = 0; i < active_obs_size; i++)
            noise(i, j) = enkf_util_rand_normal(0, 1, shared_rng);
    return noise;
}

static std::pair<Eigen::MatrixXd, analysis::ObservationHandler>
load_observations_and_responses_pybind(
    Cwrap<enkf_fs_type> source_fs, Cwrap<enkf_obs_type> obs, double alpha,
    double std_cutoff, double global_std_scaling, std::vector<bool> ens_mask,
    const std::vector<std::pair<std::string, std::vector<int>>>
        &selected_observations) {
    return analysis::load_observations_and_responses(
        source_fs, obs, alpha, std_cutoff, global_std_scaling, ens_mask,
        selected_observations);
}

} // namespace
ERT_CLIB_SUBMODULE("update", m) {
    using namespace py::literals;
    py::class_<analysis::RowScalingParameter,
               std::shared_ptr<analysis::RowScalingParameter>>(
        m, "RowScalingParameter")
        .def(py::init<std::string, std::shared_ptr<RowScaling>,
                      std::vector<int>>(),
             py::arg("name"), py::arg("row_scaling"),
             py::arg("index_list") = py::list())
        .def_readwrite("name", &analysis::RowScalingParameter::name)
        .def_readwrite("row_scaling",
                       &analysis::RowScalingParameter::row_scaling)
        .def_readonly("active_list",
                      &analysis::RowScalingParameter::active_list)
        .def_property("index_list",
                      &analysis::RowScalingParameter::get_index_list,
                      &analysis::RowScalingParameter::set_index_list)
        .def("__repr__", &::analysis::RowScalingParameter::to_string);

    py::class_<analysis::Parameter, std::shared_ptr<analysis::Parameter>>(
        m, "Parameter")
        .def(py::init<std::string, std::vector<int>>(), py::arg("name"),
             py::arg("index_list") = py::list())
        .def_readwrite("name", &analysis::Parameter::name)
        .def_readonly("active_list", &analysis::Parameter::active_list)
        .def_property("index_list", &analysis::Parameter::get_index_list,
                      &analysis::Parameter::set_index_list)
        .def("__repr__", &::analysis::Parameter::to_string);

    py::class_<analysis::ObservationHandler,
               std::shared_ptr<analysis::ObservationHandler>>(
        m, "ObservationHandler")
        .def(py::init<>())
        .def_readwrite("observation_values",
                       &analysis::ObservationHandler::observation_values,
                       py::return_value_policy::reference_internal)
        .def_readwrite("observation_errors",
                       &analysis::ObservationHandler::observation_errors,
                       py::return_value_policy::reference_internal)
        .def_readwrite("obs_mask", &analysis::ObservationHandler::obs_mask)
        .def_readwrite("update_snapshot",
                       &analysis::ObservationHandler::update_snapshot);
    m.def("load_observations_and_responses",
          load_observations_and_responses_pybind);
    m.def("save_parameter", analysis::save_parameter);
    m.def("load_parameter", analysis::load_parameter);
    m.def("generate_noise", generate_noise);
}
