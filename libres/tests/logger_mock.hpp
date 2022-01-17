#pragma once

#include <fmt/format.h>

#include <ert/logging.hpp>

class MockLogger : public ert::ILogger {
public:
    std::vector<std::string> calls;

protected:
    void log(ert::ILogger::Level level, fmt::string_view f,
             fmt::format_args args) override {
        std::string s = fmt::vformat(f, args);
        calls.push_back(s);
    }
};
