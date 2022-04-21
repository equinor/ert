#ifndef ERT_SUMMARY_KEY_MATCHER_H
#define ERT_SUMMARY_KEY_MATCHER_H

#include <ert/util/stringlist.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_types.hpp>

typedef struct summary_key_matcher_struct summary_key_matcher_type;

extern "C" summary_key_matcher_type *summary_key_matcher_alloc();
extern "C" void summary_key_matcher_free(summary_key_matcher_type *matcher);
extern "C" int
summary_key_matcher_get_size(const summary_key_matcher_type *matcher);
extern "C" void
summary_key_matcher_add_summary_key(summary_key_matcher_type *matcher,
                                    const char *summary_key);
extern "C" bool
summary_key_matcher_match_summary_key(const summary_key_matcher_type *matcher,
                                      const char *summary_key);
extern "C" bool summary_key_matcher_summary_key_is_required(
    const summary_key_matcher_type *matcher, const char *summary_key);
extern "C" stringlist_type *
summary_key_matcher_get_keys(const summary_key_matcher_type *matcher);

UTIL_IS_INSTANCE_HEADER(summary_key_matcher);

#endif
