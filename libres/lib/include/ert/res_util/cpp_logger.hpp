#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/functional.h"
#include <functional>

namespace py = pybind11;
namespace cpp_logger {

enum Levels : int { DEBUG, INFO, WARNING, ERROR };
class Logger {
    std::string name;

public:
    static std::function<void(int, std::string, std::string)> custom_callback;

    static void
    set_callback(std::function<void(int, std::string, std::string)> cb) {
        custom_callback = cb;
    }
    Logger(std::string name) : name(name) {}

    void debug(std::string msg) { callback(Levels::DEBUG, name, msg); }
    void info(std::string msg) { callback(Levels::INFO, name, msg); }
    void warning(std::string msg) { callback(Levels::WARNING, name, msg); }
    void error(std::string msg) { callback(Levels::ERROR, name, msg); }

private:
    void callback(int level, std::string name, std::string msg) {
        if (Logger::custom_callback)
            Logger::custom_callback(level, name, msg);
    }
};
} // namespace cpp_logger
