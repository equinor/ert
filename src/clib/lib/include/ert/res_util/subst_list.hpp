#ifndef ERT_SUBST_H
#define ERT_SUBST_H

#include <stdbool.h>
#include <stdio.h>

#include <ert/util/buffer.hpp>

typedef struct subst_list_struct subst_list_type;
void subst_list_fprintf(const subst_list_type *, FILE *stream);
extern "C" subst_list_type *subst_list_alloc();
extern "C" subst_list_type *subst_list_alloc_deep_copy(const subst_list_type *);
extern "C" void subst_list_free(subst_list_type *);
void subst_list_clear(subst_list_type *subst_list);
extern "C" void subst_list_append_copy(subst_list_type *, const char *,
                                       const char *);
void subst_list_append_owned_ref(subst_list_type *, const char *, const char *);

extern "C" bool subst_list_filter_file(const subst_list_type *, const char *,
                                       const char *);
extern "C" char *subst_list_alloc_filtered_string(const subst_list_type *,
                                                  const char *);
extern "C" int subst_list_get_size(const subst_list_type *);
extern "C" const char *subst_list_get_value(const subst_list_type *subst_list,
                                            const char *key);
const char *subst_list_iget_value(const subst_list_type *subst_list, int index);
extern "C" const char *subst_list_iget_key(const subst_list_type *subst_list,
                                           int index);
extern "C" bool subst_list_has_key(const subst_list_type *subst_list,
                                   const char *key);
extern "C" void subst_list_add_from_string(subst_list_type *subst_list,
                                           const char *arg_string);
#endif
