#pragma once

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <ert/abort.hpp>
#include <fmt/format.h>
#include <stdexcept>
#include <string>

/// strdup with realloc
static char *restrdup(char *old_string, const char *src) {
    if (src != nullptr) {
        size_t len = strlen(src) + 1;
        auto copy = (char *)realloc(old_string, len);
        CHECK_ALLOC(copy);
        strncpy(copy, src, len);
        return copy;
    } else {
        if (old_string != nullptr)
            free(old_string);
        return nullptr;
    }
}

/// Join a nullptr terminated arglist with space.
static std::string join_with_space(char **lsf_argv) {
    std::string result = "";
    char **argptr = lsf_argv;
    while (*argptr != nullptr) {
        result += std::string(*argptr);
        result += " ";
        argptr++;
    }
    return result;
}

static inline char *saprintf(const char *fmt, ...) {
    char *s = nullptr;
    va_list ap;
    va_start(ap, fmt);
    va_list copy_ap;
    va_copy(copy_ap, ap);
    int length = vsnprintf(nullptr, 0, fmt, copy_ap) + 1;
    va_end(copy_ap);
    s = (char *)calloc(length, 1);
    CHECK_ALLOC(s);
    int ret = vsnprintf(s, length, fmt, ap);
    va_end(ap);
    if (ret <= 0) {
        throw std::runtime_error(fmt::format("Could not print {}", fmt));
    }
    return s;
}
