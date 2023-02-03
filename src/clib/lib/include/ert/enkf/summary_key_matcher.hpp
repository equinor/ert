#ifndef ERT_SUMMARY_KEY_MATCHER_H
#define ERT_SUMMARY_KEY_MATCHER_H
#include <string>
#include <vector>

bool summary_key_matcher_match_summary_key(
    const std::vector<std::string> &summary_keys,
    const std::string &summary_key);
bool summary_key_matcher_summary_key_is_required(
    const std::vector<std::string> &summary_keys,
    const std::string &summary_key);

#endif
