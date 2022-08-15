#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <stdexcept>

/**
 * Convenience exception classes that accept arguments for fmtlib formatting
 */
namespace exc {

#define STDEXCEPT(_Name)                                                       \
    class _Name : public ::std::_Name {                                        \
    public:                                                                    \
        using ::std::_Name::_Name;                                             \
        template <typename... T>                                               \
        _Name(::fmt::format_string<T...> fmt, T &&...args)                     \
            : _Name(::fmt::format(fmt, ::std::forward<T>(args)...)) {}         \
    }

STDEXCEPT(invalid_argument);
STDEXCEPT(out_of_range);
STDEXCEPT(runtime_error);

#undef STDEXCEPT
} // namespace exc
