#include <ert/enkf/enkf_main.hpp>
#include <ert/python.hpp>

RES_LIB_SUBMODULE("enkf_fs_manager", m) {
    m.def(
        "init_current_case_from_existing_custom",
        [](py::object self, py::object source_case_py, int source_report_step,
           std::vector<std::string> &node_list, std::vector<bool> &iactive) {
            auto source_case_fs = ert::from_cwrap<enkf_fs_type>(source_case_py);
            auto enkf_main = ert::from_cwrap<enkf_main_type>(self);
            return enkf_main_init_current_case_from_existing_custom(
                enkf_main, source_case_fs, source_report_step, node_list,
                iactive);
        },
        py::arg("self"), py::arg("source_case"), py::arg("source_report_step"),
        py::arg("node_list"), py::arg("iactive"));
}
