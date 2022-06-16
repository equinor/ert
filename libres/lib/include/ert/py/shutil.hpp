#pragma once

#include <ert/python.hpp>
#include <optional>
#include <string>

namespace ertpy {

/**
 * Look up executable using Python's "shutil.which" function.
 *
 * 1. If the @ref filename is absolute (starts with '/'), then returns itself
 *    iff. it has the executable bit set
 * 2. If the @ref filename is relative (contains '/' but does not start with
 *    it), then return absolute path (from current working directory) iff. it
 *    has the executable bit set
 * 3. If the @ref filename is a command (contains no '/') then look in the
 *    PATH environment variable and return the first hit that also has the
 *    executable bit set
 *
 * @return Absolute path to the executable file or @ref std::nullopt if file is
 *         not found or is not executable
 */
inline auto which(const std::string &filename) {
    py::gil_scoped_acquire guard;

    auto shutil = py::module_::import("shutil");
    return py::cast<std::optional<std::string>>(shutil.attr("which")(filename));
}
} // namespace ertpy
