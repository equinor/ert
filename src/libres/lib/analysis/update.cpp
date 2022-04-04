#include <Eigen/Dense>
#include <assert.h>
#include <cerrno>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <vector>

#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/ies/ies.hpp>
#include <ert/analysis/ies/ies_data.hpp>
#include <ert/analysis/update.hpp>
#include <ert/enkf/enkf_analysis.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/meas_data.hpp>
#include <ert/enkf/obs_data.hpp>
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
                         const std::vector<Parameter> &parameters,
                         enkf_fs_type *target_fs,
                         const std::vector<int> &iens_active_index,
                         Eigen::MatrixXd &A) {

    int ens_size = A.cols();
    int current_row = 0;

    for (const auto &parameter : parameters) {
        const enkf_config_node_type *config_node =
            ensemble_config_get_node(ens_config, parameter.name.c_str());

        ensure_node_loaded(config_node, target_fs);
        int active_size = parameter.active_list.active_size(
            enkf_config_node_get_data_size(config_node, 0));

        if ((active_size + current_row) > A.rows())
            A.conservativeResize(A.rows() + 2 * active_size, ens_size);
        if (active_size > 0) {
            for (int column = 0; column < ens_size; column++) {
                int iens = iens_active_index[column];
                serialize_node(target_fs, config_node, iens, current_row,
                               column, &parameter.active_list, A);
            }
            current_row += active_size;
        }
    }
    A.conservativeResize(current_row, ens_size);
}

void deserialize_node(enkf_fs_type *fs,
                      const enkf_config_node_type *config_node, int iens,
                      int row_offset, int column, const ActiveList *active_list,
                      const Eigen::MatrixXd &A) {

    node_id_type node_id = {.report_step = 0, .iens = iens};
    enkf_node_type *node = enkf_node_alloc(config_node);

    // If partly active (deserialize will fill it only in part)
    enkf_node_load(node, fs, node_id);

    // deserialize the matrix into the node (and writes it to the fs)
    enkf_node_deserialize(node, fs, node_id, active_list, A, row_offset,
                          column);
    enkf_fs_get_state_map(fs).update_undefined(iens, STATE_INITIALIZED);
    enkf_node_free(node);
}

void assert_matrix_size(const Eigen::MatrixXd &m, const char *name, int rows,
                        int columns) {
    if (!((m.rows() == rows) && (m.cols() == columns)))
        throw std::invalid_argument("matrix mismatch " + std::string(name) +
                                    ":[" + std::to_string(m.rows()) + "," +
                                    std::to_string(m.cols()) +
                                    "   - expected:[" + std::to_string(rows) +
                                    "," + std::to_string(columns) + "]");
}

/**
load a set of parameters from a enkf_fs_type storage into a set of
matrices.
*/
std::optional<Eigen::MatrixXd>
load_parameters(enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
                const std::vector<int> &iens_active_index,
                const std::vector<Parameter> &parameters) {

    int active_ens_size = iens_active_index.size();
    if (!parameters.empty()) {
        int matrix_start_size = 250000;
        Eigen::MatrixXd A =
            Eigen::MatrixXd::Zero(matrix_start_size, active_ens_size);

        serialize_parameter(ensemble_config, parameters, target_fs,
                            iens_active_index, A);
        return A;
    }

    return {};
}

void save_parameters(enkf_fs_type *target_fs,
                     ensemble_config_type *ensemble_config,
                     const std::vector<int> &iens_active_index,
                     const std::vector<Parameter> &parameters,
                     const Eigen::MatrixXd &A) {

    int ens_size = iens_active_index.size();
    int current_row = 0;
    for (auto &parameter : parameters) {
        const enkf_config_node_type *config_node =
            ensemble_config_get_node(ensemble_config, parameter.name.c_str());
        ensure_node_loaded(config_node, target_fs);
        int active_size = parameter.active_list.active_size(
            enkf_config_node_get_data_size(config_node, 0));
        if (active_size > 0) {
            for (int column = 0; column < ens_size; column++) {
                int iens = iens_active_index[column];
                deserialize_node(target_fs, config_node, iens, current_row,
                                 column, &parameter.active_list, A);
            }
            current_row += active_size;
        }
    }
}

/**
Store a parameters into a enkf_fs_type storage
*/
void save_row_scaling_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    const std::vector<int> &iens_active_index,
    const std::vector<RowScalingParameter> &scaled_parameters,
    const std::vector<std::pair<Eigen::MatrixXd, std::shared_ptr<RowScaling>>>
        &scaled_A) {
    if (scaled_A.size() > 0) {
        int ikw = 0;
        for (auto &scaled_parameter : scaled_parameters) {
            auto &A = scaled_A[ikw].first;
            for (int column = 0; column < iens_active_index.size(); column++) {
                int iens = iens_active_index[column];
                deserialize_node(
                    target_fs,
                    ensemble_config_get_node(ensemble_config,
                                             scaled_parameter.name.c_str()),
                    iens, 0, column, &scaled_parameter.active_list, A);
            }
            ikw++;
        }
    }
}

/**
load a set of parameters from a enkf_fs_type storage into a set of
matrices with the corresponding row-scaling object.
*/
std::vector<std::pair<Eigen::MatrixXd, std::shared_ptr<RowScaling>>>
load_row_scaling_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    const std::vector<int> &iens_active_index,
    const std::vector<RowScalingParameter> &config_parameters) {

    std::vector<std::pair<Eigen::MatrixXd, std::shared_ptr<RowScaling>>>
        parameters;
    int active_ens_size = iens_active_index.size();
    if (!config_parameters.empty()) {
        int matrix_start_size = 250000;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(250000, active_ens_size);

        for (const auto &parameter : config_parameters) {
            const auto *config_node = ensemble_config_get_node(
                ensemble_config, parameter.name.c_str());
            const int node_size =
                enkf_config_node_get_data_size(config_node, 0);
            if (A.rows() < node_size)
                A.conservativeResize(node_size, active_ens_size);
            for (int column = 0; column < iens_active_index.size(); column++) {
                int iens = iens_active_index[column];
                serialize_node(target_fs, config_node, iens, 0, column,
                               &parameter.active_list, A);
            }
            auto row_scaling = parameter.row_scaling;

            A.conservativeResize(row_scaling->size(), A.cols());
            parameters.emplace_back(std::move(A), row_scaling);
        }
    }

    return parameters;
}

/**
Copy all parameters from source_fs to target_fs
*/
void copy_parameters(enkf_fs_type *source_fs, enkf_fs_type *target_fs,
                     const ensemble_config_type *ensemble_config,
                     const std::vector<bool> &ens_mask) {

    /*
      Copy all the parameter nodes from source case to target case;
      nodes which are updated will be fetched from the new target
      case, and nodes which are not updated will be manually copied
      over there.
    */
    if (target_fs != source_fs) {
        std::vector<int> ens_active_list = bool_vector_to_active_list(ens_mask);
        std::vector<std::string> param_keys =
            ensemble_config_keylist_from_var_type(ensemble_config, PARAMETER);
        for (auto &key : param_keys) {
            enkf_config_node_type *config_node =
                ensemble_config_get_node(ensemble_config, key.c_str());
            enkf_node_type *data_node = enkf_node_alloc(config_node);
            for (int j : ens_active_list) {
                node_id_type node_id;
                node_id.iens = j;
                node_id.report_step = 0;

                enkf_node_load(data_node, source_fs, node_id);
                enkf_node_store(data_node, target_fs, node_id);
            }
            enkf_node_free(data_node);
        }

        auto &target_state_map = enkf_fs_get_state_map(target_fs);
        target_state_map.set_from_inverted_mask(ens_mask, STATE_PARENT_FAILURE);
        target_state_map.set_from_mask(ens_mask, STATE_INITIALIZED);
        enkf_fs_fsync(target_fs);
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

    return std::pair<Eigen::MatrixXd, ObservationHandler>(
        S, ObservationHandler(observation_values, observation_errors, obs_mask,
                              update_snapshot));
}
} // namespace analysis

namespace {
static Eigen::MatrixXd generate_noise(int active_obs_size, int active_ens_size,
                                      py::object shared_rng) {
    auto shared_rng_ = ert::from_cwrap<rng_type>(shared_rng);
    Eigen::MatrixXd noise =
        Eigen::MatrixXd::Zero(active_obs_size, active_ens_size);
    for (int j = 0; j < active_ens_size; j++)
        for (int i = 0; i < active_obs_size; i++)
            noise(i, j) = enkf_util_rand_normal(0, 1, shared_rng_);
    return noise;
}

static void copy_parameters_pybind(py::object source_fs, py::object target_fs,
                                   py::object ensemble_config,
                                   std::vector<bool> ens_mask) {
    auto ensemble_config_ =
        ert::from_cwrap<ensemble_config_type>(ensemble_config);
    auto source_fs_ = ert::from_cwrap<enkf_fs_type>(source_fs);
    auto target_fs_ = ert::from_cwrap<enkf_fs_type>(target_fs);
    return analysis::copy_parameters(source_fs_, target_fs_, ensemble_config_,
                                     ens_mask);
}

static std::pair<Eigen::MatrixXd, analysis::ObservationHandler>
load_observations_and_responses_pybind(
    py::object source_fs, py::object obs, double alpha, double std_cutoff,
    double global_std_scaling, std::vector<bool> ens_mask,
    const std::vector<std::pair<std::string, std::vector<int>>>
        &selected_observations) {

    auto source_fs_ = ert::from_cwrap<enkf_fs_type>(source_fs);
    auto obs_ = ert::from_cwrap<enkf_obs_type>(obs);

    return analysis::load_observations_and_responses(
        source_fs_, obs_, alpha, std_cutoff, global_std_scaling, ens_mask,
        selected_observations);
}

static std::vector<std::pair<Eigen::MatrixXd, std::shared_ptr<RowScaling>>>
load_row_scaling_parameters_pybind(
    py::object target_fs, py::object ensemble_config,
    const std::vector<int> &iens_active_index,
    const std::vector<analysis::RowScalingParameter> &config_parameters) {

    auto target_fs_ = ert::from_cwrap<enkf_fs_type>(target_fs);
    auto ensemble_config_ =
        ert::from_cwrap<ensemble_config_type>(ensemble_config);

    return analysis::load_row_scaling_parameters(
        target_fs_, ensemble_config_, iens_active_index, config_parameters);
}

static std::optional<Eigen::MatrixXd>
load_parameters_pybind(py::object target_fs, py::object ensemble_config,
                       const std::vector<int> &iens_active_index,
                       const std::vector<analysis::Parameter> &parameters) {

    auto target_fs_ = ert::from_cwrap<enkf_fs_type>(target_fs);
    auto ensemble_config_ =
        ert::from_cwrap<ensemble_config_type>(ensemble_config);

    return analysis::load_parameters(target_fs_, ensemble_config_,
                                     iens_active_index, parameters);
}

static void save_parameters_pybind(py::object target_fs,
                                   py::object ensemble_config,
                                   std::vector<int> iens_active_index,
                                   std::vector<analysis::Parameter> &parameters,
                                   const Eigen::MatrixXd &A) {
    auto target_fs_ = ert::from_cwrap<enkf_fs_type>(target_fs);
    auto ensemble_config_ =
        ert::from_cwrap<ensemble_config_type>(ensemble_config);

    analysis::save_parameters(target_fs_, ensemble_config_, iens_active_index,
                              parameters, A);
}
static void save_row_scaling_parameters_pybind(
    py::object target_fs, py::object ensemble_config,
    std::vector<int> iens_active_index,
    const std::vector<analysis::RowScalingParameter> &config_parameters,
    const std::vector<std::pair<Eigen::MatrixXd, std::shared_ptr<RowScaling>>>
        scaled_A) {
    auto target_fs_ = ert::from_cwrap<enkf_fs_type>(target_fs);
    auto ensemble_config_ =
        ert::from_cwrap<ensemble_config_type>(ensemble_config);

    analysis::save_row_scaling_parameters(target_fs_, ensemble_config_,
                                          iens_active_index, config_parameters,
                                          scaled_A);
}

} // namespace
RES_LIB_SUBMODULE("update", m) {
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
    m.def("copy_parameters", copy_parameters_pybind);
    m.def("load_observations_and_responses",
          load_observations_and_responses_pybind);
    m.def("save_parameters", save_parameters_pybind);
    m.def("save_row_scaling_parameters", save_row_scaling_parameters_pybind);
    m.def("load_parameters", load_parameters_pybind);
    m.def("load_row_scaling_parameters", load_row_scaling_parameters_pybind);
    m.def("generate_noise", generate_noise);
}
