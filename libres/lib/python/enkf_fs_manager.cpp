#include <ert/enkf/enkf_main.hpp>
#include <ert/python.hpp>

RES_LIB_SUBMODULE("enkf_fs_manager", m) {
    m.def(
        "initialize_from_scratch",
        [](py::object self, std::vector<std::string> param_list,
           py::object run_context_py) {
            auto enkf_main = ert::from_cwrap<enkf_main_type>(self);
            auto run_context =
                ert::from_cwrap<ert_run_context_type>(run_context_py);
            return enkf_main_initialize_from_scratch(enkf_main, param_list,
                                                     run_context);
        },
        py::arg("self"), py::arg("param_list"), py::arg("run_context"));
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
