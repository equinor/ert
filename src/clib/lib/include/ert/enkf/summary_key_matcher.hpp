#ifndef ERT_SUMMARY_KEY_MATCHER_H
#define ERT_SUMMARY_KEY_MATCHER_H

#include <ert/util/stringlist.h>

#include <ert/enkf/enkf_types.hpp>

typedef struct summary_key_matcher_struct summary_key_matcher_type;

summary_key_matcher_type *summary_key_matcher_alloc();
void summary_key_matcher_free(summary_key_matcher_type *matcher);
int summary_key_matcher_get_size(const summary_key_matcher_type *matcher);
void summary_key_matcher_add_summary_key(summary_key_matcher_type *matcher,
                                         const char *summary_key);
bool summary_key_matcher_match_summary_key(
    const summary_key_matcher_type *matcher, const char *summary_key);
bool summary_key_matcher_summary_key_is_required(
    const summary_key_matcher_type *matcher, const char *summary_key);
stringlist_type *
summary_key_matcher_get_keys(const summary_key_matcher_type *matcher);

#endif
