#include <ert/config/config_settings.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/python.hpp>

ERT_CLIB_SUBMODULE("config_keywords", m) {
    using namespace py::literals;
    m.def(
        "init_res_config_parser",
        [](py::object py_config_parser) {
            auto config_parser =
                ert::from_cwrap<config_parser_type>(py_config_parser);
            model_config_init_config_parser(config_parser);
            config_add_key_value(config_parser, RES_CONFIG_FILE_KEY, false,
                                 CONFIG_EXISTING_PATH);
            config_add_key_value(config_parser, CONFIG_DIRECTORY_KEY, false,
                                 CONFIG_EXISTING_PATH);
        },
        py::arg("config_parser"));
}
