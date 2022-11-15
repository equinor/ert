#include "ert/enkf/ensemble_config.hpp"
#include <Eigen/Dense>
#include <cerrno>
#include <fmt/format.h>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <ert/analysis/update.hpp>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_obs.hpp>
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
    try {
        enkf_node_serialize(node, fs, node_id, active_list, A, row_offset,
                            column);
    } catch (const std::out_of_range &) {
        std::string param_name = enkf_node_get_key(node);
        enkf_node_free(node);
        throw pybind11::key_error(
            fmt::format("No parameter: {} in storage", param_name));
    }
    enkf_node_serialize(node, fs, node_id, active_list, A, row_offset, column);
    enkf_node_free(node);
}

void serialize_parameter(const enkf_config_node_type *config_node,
                         const Parameter parameter, enkf_fs_type *target_fs,
                         const std::vector<int> &iens_active_index,
                         Eigen::MatrixXd &A) {

    int ens_size = A.cols();

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
                               Cwrap<enkf_config_node_type> enkf_config_node,
                               const std::vector<int> &iens_active_index,
                               const Parameter parameter) {

    int active_ens_size = iens_active_index.size();
    int matrix_start_size = 250000;
    Eigen::MatrixXd A =
        Eigen::MatrixXd::Zero(matrix_start_size, active_ens_size);

    serialize_parameter(enkf_config_node, parameter, source_fs,
                        iens_active_index, A);
    return A;
}

/**
save a paramater matrix to enkf_fs_type storage
*/
void save_parameter(Cwrap<enkf_fs_type> target_fs,
                    Cwrap<enkf_config_node_type> config_node,
                    const std::vector<int> &iens_active_index,
                    const Parameter parameter, const Eigen::MatrixXd &A) {

    int ens_size = iens_active_index.size();

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

} // namespace analysis

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

    m.def("save_parameter", analysis::save_parameter);
    m.def("load_parameter", analysis::load_parameter);
}
