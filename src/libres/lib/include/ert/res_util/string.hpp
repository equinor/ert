#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace ert {

/**
 * Non memory allocating string splitting function
 *
 * @note The input string is a view. It is therefore Undefined Behaviour to
 * modify the original string from inside the `func` parameter.
 *
 * @param[in] str String to be split
 * @param[in] delimiter Delimiter to split the string by
 * @param func Callback which is called with the string parts
 */
template <typename Func>
void split(std::string_view str, char delimiter, Func &&func) {
    size_t pos = 0, next_pos = 0;

    if (str.empty())
        return;

    while ((next_pos = str.find(delimiter, pos)) != str.npos) {
        func(str.substr(pos, next_pos - pos));
        pos = next_pos + 1;
    }
    func(str.substr(pos));
}

/**
 * Split string into a container of strings
 *
 * @param[in] str String to be split
 * @param[in] delimiter Delimiter to split the string by
 * @return List of the parts
 */
inline std::vector<std::string> split(std::string_view str, char delimiter) {
    std::vector<std::string> vec;
    split(str, delimiter,
          [&vec](std::string_view substr) { vec.emplace_back(substr); });
    return vec;
}

/**
 * Split string and get the back element
 *
 * Equivalent to Python's: `s.split(delim)[-1]`
 */
inline std::string_view back_element(std::string_view str, char delimiter) {
    auto pos = str.rfind(delimiter);
    return pos == str.npos ? str : str.substr(pos + 1);
}

/**
 * Join a ForwardIterable container with separator
 */
template <typename ForwardIterable>
std::string join(const ForwardIterable &container, std::string_view separator) {
    // Require that container contains strings
    using value_type = typename ForwardIterable::value_type;
    static_assert(std::is_same_v<value_type, const char *> ||
                      std::is_same_v<value_type, std::string> ||
                      std::is_same_v<value_type, std::string_view>,
                  "'container' must contain string types");

    auto it = container.cbegin();
    auto end = container.cend();

    std::string joined;

    if (it == end) {
        // Empty container, return empty string
        return joined;
    }

    // Append first element
    joined.append(*it);
    ++it;

    // Append the rest
    for (; it != end; ++it) {
        joined.append(separator);
        joined.append(*it);
    }
    return joined;
}
} // namespace ert
