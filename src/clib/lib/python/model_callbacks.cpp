#include <ert/enkf/ecl_config.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/enkf/run_arg.hpp>
#include <ert/python.hpp>

ERT_CLIB_SUBMODULE("model_callbacks", m) {
    py::enum_<fw_load_status>(m, "LoadStatus", py::arithmetic())
        .value("LOAD_SUCCESSFUL", LOAD_SUCCESSFUL)
        .value("LOAD_FAILURE", LOAD_FAILURE)
        .value("TIME_MAP_FAILURE", TIME_MAP_FAILURE)
        .export_values();

    m.def("forward_model_exit", [](std::vector<py::object> arr) {
        auto run_arg = ert::from_cwrap<run_arg_type>(arr[0]);
        return enkf_state_complete_forward_model_EXIT_handler__(run_arg);
    });

    m.def("forward_model_ok", [](py::object run_arg, py::object ens_conf,
                                 py::object model_conf, py::object ecl_conf) {
        auto result = enkf_state_load_from_forward_model(
            ert::from_cwrap<ensemble_config_type>(ens_conf),
            ert::from_cwrap<model_config_type>(model_conf),
            ert::from_cwrap<ecl_config_type>(ecl_conf),
            ert::from_cwrap<run_arg_type>(run_arg));

        if (result.first == LOAD_SUCCESSFUL) {
            result.second = "Results loaded successfully.";
        }

        return result;
    });
}
