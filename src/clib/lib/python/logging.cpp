#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ert/python.hpp>

using namespace std::string_literals;
namespace py = pybind11;

// Controls the prefix of the Python logger:
//   logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")
#define LOGGER_NAMESPACE "ert"s

namespace {
class Logger : public ert::ILogger {
    struct interface {
        py::object debug;
        py::object info;
        py::object warning;
        py::object error;
        py::object critical;

        interface(py::object logger);
    };

    std::unique_ptr<interface> m_interface;

public:
    void init(const std::string &name);

    /** Called on Python shutdown to release the py::objects */
    void shutdown() { m_interface = nullptr; }

protected:
    void log(Level level, fmt::string_view f, fmt::format_args args) final;
};

bool has_init_logging{};

/*
 * Workaround for C++'s dynamic initialisation. The initialisation order is
 * top-to-bottom in a single translation unit (C++ source file), but unspecified
 * across multiple TUs. As such, ert::get_logger might get called before the
 * std::unordered_map is initialised, which leads to a segfault or similar.
 * However, static variables in a function will get lazily evaluated during call
 * time, thus guaranteeing that the std::unordered_map is ready to use.
 */
auto &loggers() {
    static std::unordered_map<std::string, std::shared_ptr<Logger>> map;
    return map;
}
} // namespace

/**
 * Fetch functions from Python. Note that we keep hold of the functions, but
 * they can still get monkeypatched by eg. pytest's caplog.
 */
Logger::interface::interface(py::object logger)
    : debug(logger.attr("debug")), info(logger.attr("info")),
      warning(logger.attr("warning")), error(logger.attr("error")),
      critical(logger.attr("critical")) {}

void Logger::init(const std::string &name) {
    // Calling via cwrap does not acquire GIL, so do it now
    py::gil_scoped_acquire gil;

    auto get_logger = py::module_::import("logging").attr("getLogger");

    // Don't include the . for the default logger (whose name is "")
    auto full_name =
        name.empty() ? LOGGER_NAMESPACE : LOGGER_NAMESPACE "."s + name;
    auto logger = get_logger(full_name);
    m_interface = std::make_unique<interface>(logger);
}

void Logger::log(Logger::Level level, fmt::string_view f,
                 fmt::format_args args) {
    if (!m_interface) {
        // Python has either not initalised us yet or has shut down. Drop logs.
        return;
    }

    // Calling via cwrap does not acquire GIL, so do it now
    py::gil_scoped_acquire gil;

    auto payload = fmt::vformat(f, args);
    switch (level) {
    case Level::debug:
        m_interface->debug(payload);
        break;
    case Level::info:
        m_interface->info(payload);
        break;
    case Level::warning:
        m_interface->warning(payload);
        break;
    case Level::error:
        m_interface->error(payload);
        break;
    case Level::critical:
        m_interface->critical(payload);
        break;
    }
}

/* Declare weak so it's possible to override in tests */
[[gnu::weak]] std::shared_ptr<ert::ILogger>
ert::get_logger(const std::string &name) {
    auto it = loggers().find(name);
    if (it != loggers().end())
        return it->second;

    auto logger = std::make_shared<Logger>();
    loggers()[name] = logger;

    if (has_init_logging)
        logger->init(name);

    return logger;
}

ERT_CLIB_SUBMODULE("", m) {
    has_init_logging = true;

    // Initialise all loggers that were created before Python initialised
    for (auto &[name, logger] : loggers()) {
        logger->init(name);
    }

    // For testing purposes, add a function which will log to all levels
    m.def("_test_logger", [](const std::string &str) {
        auto logger = ert::get_logger("_test_logger");

        logger->debug("debug: {}", str);
        logger->info("info: {}", str);
        logger->warning("warning: {}", str);
        logger->error("error: {}", str);
        logger->critical("critical: {}", str);
    });

    // Add a cleanup routine for when Python attemps to release the module
    // during shutdown. Release the py::objects held by Logger.
    auto cleanup = [] {
        has_init_logging = false;
        for (auto &[name, logger] : loggers()) {
            logger->shutdown();
        }
    };
    m.add_object("_cleanup_logging", py::capsule(cleanup));
}
