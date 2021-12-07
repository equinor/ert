#include <ert/res_util/cpp_logger.hpp>

void init_logger(py::module_ m) {
    py::class_<cpp_logger::Logger>(m, "_Logger")
        .def_static("set_callback", &cpp_logger::Logger::set_callback);

    m.add_object("__cleanup_Logger", py::capsule([]() {
                     cpp_logger::Logger::set_callback(nullptr);
                 }));

    py::enum_<cpp_logger::Levels>(m, "_Levels")
        .value("DEBUG", cpp_logger::Levels::DEBUG)
        .value("INFO", cpp_logger::Levels::INFO)
        .value("WARNING", cpp_logger::Levels::WARNING)
        .value("ERROR", cpp_logger::Levels::ERROR)
        .export_values();
}

std::function<void(int level, std::string, std::string)>
    cpp_logger::Logger::custom_callback = nullptr;
