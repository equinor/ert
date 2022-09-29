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

    m.def("forward_model_exit", [](py::sequence arr) {
        auto run_arg = ert::from_cwrap<run_arg_type>(arr[0]);
        return enkf_state_complete_forward_model_EXIT_handler__(run_arg);
    });

    m.def("forward_model_ok", [](Cwrap<run_arg_type> run_arg,
                                 Cwrap<ensemble_config_type> ens_conf,
                                 Cwrap<model_config_type> model_conf) {
        auto result =
            enkf_state_load_from_forward_model(ens_conf, model_conf, run_arg);

        if (result.first == LOAD_SUCCESSFUL) {
            result.second = "Results loaded successfully.";
        }

        return result;
    });
}
