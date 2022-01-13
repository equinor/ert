#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <ert/logging.hpp>

namespace ert {
/**
 * Reset all entries in the mocked ert::ILogger
 */
void reset_loggers();

/**
 * Fetch the entries of a particular logger
 *
 * @param name The name of the logger
 * @returns Entries as a sequence of <LogLevel, Message>
 * @throws std::logic_error If the @ref name logger does not exist
 */
const std::vector<std::pair<ert::ILogger::Level, std::string>> &
get_logger_entries(const std::string &name);
} // namespace ert
