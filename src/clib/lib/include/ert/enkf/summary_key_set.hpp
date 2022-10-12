#ifndef ERT_SUMMARY_KEY_SET_H
#define ERT_SUMMARY_KEY_SET_H

#include <ert/enkf/enkf_types.hpp>
#include <ert/util/stringlist.hpp>

typedef struct summary_key_set_struct summary_key_set_type;

extern "C" summary_key_set_type *summary_key_set_alloc();
extern "C" summary_key_set_type *
summary_key_set_alloc_from_file(const char *filename, bool read_only);
extern "C" void summary_key_set_free(summary_key_set_type *set);
extern "C" int summary_key_set_get_size(summary_key_set_type *set);
extern "C" bool summary_key_set_add_summary_key(summary_key_set_type *set,
                                                const char *summary_key);
extern "C" PY_USED bool
summary_key_set_has_summary_key(summary_key_set_type *set,
                                const char *summary_key);
extern "C" stringlist_type *
summary_key_set_alloc_keys(summary_key_set_type *set);
extern "C" bool summary_key_set_is_read_only(const summary_key_set_type *set);

extern "C" void summary_key_set_fwrite(summary_key_set_type *set,
                                       const char *filename);
bool summary_key_set_fread(summary_key_set_type *set, const char *filename);

#endif
