#include <ert/enkf/summary_key_matcher.hpp>

#include <algorithm>
#include <fnmatch.h>
#include <iterator>

bool summary_key_matcher_match_summary_key(
    const std::vector<std::string> &summary_keys,
    const std::string &summary_key) {
    for (auto pattern : summary_keys) {
        if (fnmatch(pattern.c_str(), summary_key.c_str(), 0) == 0) {
            return true;
        }
    }
    return false;
}

bool summary_key_matcher_summary_key_is_required(
    const std::vector<std::string> &summary_keys,
    const std::string &summary_key) {
    if (!summary_key.find('*')) {
        for (auto key : summary_keys) {
            if (key == summary_key)
                return true;
        }
    }
    return false;
}
