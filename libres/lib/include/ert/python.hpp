#ifndef RES_PYTHON_HPP
#define RES_PYTHON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

/**
 * This header contains utilities for interacting with Python via pybind11
 */
namespace ert {
/**
 * Logger which forwards to Python
 */
class ILogger {
public:
    enum struct Level { debug, info, warning, error, critical };
    virtual ~ILogger() = default;

    template <typename... Args> void debug(fmt::string_view f, Args &&...args) {
        this->log(Level::debug, f,
                  fmt::make_format_args(std::forward<Args>(args)...));
    }

    template <typename... Args> void info(fmt::string_view f, Args &&...args) {
        this->log(Level::info, f,
                  fmt::make_format_args(std::forward<Args>(args)...));
    }

    template <typename... Args>
    void warning(fmt::string_view f, Args &&...args) {
        this->log(Level::warning, f,
                  fmt::make_format_args(std::forward<Args>(args)...));
    }

    template <typename... Args> void error(fmt::string_view f, Args &&...args) {
        this->log(Level::error, f,
                  fmt::make_format_args(std::forward<Args>(args)...));
    }

    template <typename... Args>
    void critical(fmt::string_view f, Args &&...args) {
        this->log(Level::critical, f,
                  fmt::make_format_args(std::forward<Args>(args)...));
    }

protected:
    virtual void log(Level level, fmt::string_view f,
                     fmt::format_args args) = 0;
};

/**
 * Create or get a Logger which syncs to Python
 *
 * Creates a logger that logs only to Python's logger of the same
 * name, prefixed with libres' namespace. Can be created statically, although
 * it no-ops outside of the lifetime of this Python module.
 *
 * That is, this logger will only log after this module's Python init function
 * is called, and before Python shuts down, but is valid for all other times.
 */
std::shared_ptr<ILogger> get_logger(const std::string &name);
} // namespace ert

#endif
