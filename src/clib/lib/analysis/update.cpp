#include <Eigen/Dense>
#include <cerrno>
#include <fmt/format.h>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <ert/analysis/update.hpp>
#include <ert/except.hpp>
#include <ert/python.hpp>
#include <ert/res_util/memory.hpp>
#include <ert/res_util/metric.hpp>

static auto logger = ert::get_logger("analysis.update");

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
}
