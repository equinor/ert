#include <ert/enkf/enkf_main.hpp>
#include <ert/python.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void enkf_fs_manager(py::module_ m) {
    m.def(
        "initialize_from_scratch",
        [](py::object self, std::vector<std::string> param_list,
           py::object run_context_py) {
            auto enkf_main = py_conversion::to_enkf_main_type(self);
            auto run_context =
                py_conversion::to_run_context_type(run_context_py);
            return enkf_main_initialize_from_scratch(enkf_main, param_list,
                                                     run_context);
        },
        py::arg("self"), py::arg("param_list"), py::arg("run_context"));
    m.def(
        "init_current_case_from_existing_custom",
        [](py::object self, py::object source_case_py, int source_report_step,
           std::vector<std::string> &node_list, std::vector<bool> &iactive) {
            auto source_case_fs =
                py_conversion::to_enkf_fs_type(source_case_py);
            auto enkf_main = py_conversion::to_enkf_main_type(self);
            return enkf_main_init_current_case_from_existing_custom(
                enkf_main, source_case_fs, source_report_step, node_list,
                iactive);
        },
        py::arg("self"), py::arg("source_case"), py::arg("source_report_step"),
        py::arg("node_list"), py::arg("iactive"));
}
