#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

/// strdup with realloc
static char *restrdup(char *old_string, const char *src) {
    if (src != nullptr) {
        size_t len = strlen(src) + 1;
        auto copy = (char *)realloc(old_string, len);
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
