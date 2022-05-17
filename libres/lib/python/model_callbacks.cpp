#include <ert/enkf/res_config.hpp>
#include <ert/enkf/run_arg.hpp>
#include <ert/python.hpp>

RES_LIB_SUBMODULE("model_callbacks", m) {
    m.def("forward_model_exit", [](std::vector<py::object> arr) {
        auto run_arg = ert::from_cwrap<run_arg_type>(arr[0]);
        return enkf_state_complete_forward_model_EXIT_handler__(run_arg);
    });

    m.def("forward_model_ok", [](std::vector<py::object> arr) {
        auto run_arg = ert::from_cwrap<run_arg_type>(arr[0]);
        const auto res_conf = ert::from_cwrap<res_config_type>(arr[1]);
        return enkf_state_complete_forward_modelOK(res_conf, run_arg);
    });
}
