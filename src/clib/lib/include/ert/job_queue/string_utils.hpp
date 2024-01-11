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

/**
  This function parses a string literal  containing a
  represantation of a double. The return value is true|false depending
  on the success of the parse operation, the parsed value is returned
  by reference.


  Example:
  --------
  const char * s = "78.92"
  double value;

  if (sscanf_double(s , &value))
    printf("%s is a valid double\n");
  else
    printf("%s is NOT a valid double\n");

*/
static inline bool sscanf_double(const char *buffer, double *value) {
    if (!buffer)
        return false;

    bool value_OK = false;
    char *error_ptr;

    double tmp_value = strtod(buffer, &error_ptr);
    // Skip trailing white-space
    while (error_ptr[0] != '\0' && isspace(error_ptr[0]))
        error_ptr++;

    if (error_ptr[0] == '\0') {
        value_OK = true;
        if (value != NULL)
            *value = tmp_value;
    }
    return value_OK;
}

/**
   Takes a char buffer as input, and parses it as an integer. Returns
   true if the parsing succeeded, and false otherwise. If parsing
   succeeded, the integer value is returned by reference.
*/
static inline bool sscanf_int(const char *buffer, int *value) {
    if (!buffer)
        return false;

    bool value_OK = false;
    char *error_ptr;

    int tmp_value = strtol(buffer, &error_ptr, 10);

    // Skip trailing white-space
    while (error_ptr[0] != '\0' && isspace(error_ptr[0]))
        error_ptr++;

    if (error_ptr[0] == '\0') {
        value_OK = true;
        if (value != NULL)
            *value = tmp_value;
    }
    return value_OK;
}

static bool fseek_string(FILE *stream, const char *string, bool skip_string) {
    bool string_found = false;

    size_t len = strlen(string);
    long initial_pos = ftell(stream);
    if (initial_pos == -1L)
        throw std::runtime_error(
            fmt::format("ftell failed: %d/%s \n", errno, strerror(errno)));
    bool cont = true;
    do {
        int c = fgetc(stream);

        if (c == string[0]) {
            /* we got the first character right - lets try in more detail: */
            long current_pos = ftell(stream);
            if (current_pos == -1L)
                throw std::runtime_error(fmt::format("ftell failed: %d/%s \n",
                                                     errno, strerror(errno)));
            bool equal = true;
            for (int string_index = 1; string_index < len; string_index++) {
                c = fgetc(stream);

                if (c != string[string_index]) {
                    equal = false;
                    break;
                }
            }

            if (equal) {
                string_found = true;
                cont = false;
            } else /* Go back to current pos and continue searching. */
                if (fseek(stream, current_pos, SEEK_SET) != 0)
                    throw std::runtime_error(fmt::format(
                        "fseek failed: %d/%s \n", errno, strerror(errno)));
        }
        if (c == EOF)
            cont = false;
    } while (cont);

    if (string_found) {
        if (!skip_string) {
            long offset = (long)strlen(string);
            if (fseek(stream, -offset, SEEK_CUR) != 0)
                throw std::runtime_error(fmt::format("fseek failed: %d/%s \n",
                                                     errno, strerror(errno)));
        }
    } else
        // Could not find the string reposition at initial position
        if (fseek(stream, initial_pos, SEEK_SET) != 0)
            throw std::runtime_error(
                fmt::format("fseek failed: %d/%s \n", errno, strerror(errno)));
    return string_found;
}

/**
  This function will allocate a character buffer, and read file
  content all the way up to 'stop_string'. If the stop_string is not
  found, the function will return NULL, and the file pointer will be
  unchanged.
*/
static char *fscanf_upto(FILE *stream, const char *stop_string) {
    long start_pos = ftell(stream);
    if (start_pos == -1L)
        throw std::runtime_error(
            fmt::format("ftell failed: %d/%s \n", errno, strerror(errno)));
    if (fseek_string(stream, stop_string,
                     false)) { /* Default case sensitive. */
        long end_pos = ftell(stream);
        if (end_pos == -1L)
            throw std::runtime_error(
                fmt::format("ftell failed: %d/%s \n", errno, strerror(errno)));
        int len = end_pos - start_pos;
        char *buffer = (char *)calloc((len + 1), sizeof *buffer);
        CHECK_ALLOC(buffer);

        if (fseek(stream, start_pos, SEEK_SET) != 0) {
            free(buffer);
            throw std::runtime_error(
                fmt::format("fseek failed: %d/%s \n", errno, strerror(errno)));
        }
        size_t items_read = fread(buffer, 1, len, stream);
        if (items_read != len) {
            free(buffer);
            throw std::runtime_error(
                fmt::format("Could not read {} in bsub output", stop_string));
        }

        buffer[len] = '\0';

        return buffer;
    } else
        return NULL; /* stop_string not found */
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
