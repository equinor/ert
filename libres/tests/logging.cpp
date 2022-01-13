#include <unordered_map>
#include <memory>
#include <stdexcept>
#include "ert/logging.hpp"
#include "logging.hpp"

namespace {
class MockLogger : public ert::ILogger {
public:
    std::vector<std::pair<Level, std::string>> lines;

    void log(Level level, fmt::string_view f, fmt::format_args args) override {
        lines.emplace_back(level, fmt::vformat(f, args));
    }
};

auto &loggers() {
    static std::unordered_map<std::string, std::shared_ptr<MockLogger>> map;
    return map;
}
} // namespace

/* Overload the [[gnu::weak]] function defined in libres.so */
std::shared_ptr<ert::ILogger> ert::get_logger(const std::string &name) {
    auto it = loggers().find(name);
    if (it != loggers().end())
        return it->second;

    auto logger = std::make_shared<MockLogger>();
    loggers()[name] = logger;

    return logger;
}

void ert::reset_loggers() {
    for (auto &[_key, val] : loggers()) {
        val->lines.clear();
    }
}

const std::vector<std::pair<ert::ILogger::Level, std::string>> &
ert::get_logger_entries(const std::string &name) {
    auto it = loggers().find(name);
    if (it != loggers().end())
        return it->second->lines;
    throw std::logic_error(fmt::format("Logger '{}' not found", name));
}
