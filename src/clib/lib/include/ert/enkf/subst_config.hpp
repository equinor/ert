#ifndef ERT_SUBST_CONFIG_H
#define ERT_SUBST_CONFIG_H

#include <ert/tooling.hpp>

typedef struct subst_config_struct subst_config_type;

extern "C" subst_config_type *
subst_config_alloc(const config_content_type *user_config);
extern "C" PY_USED subst_config_type *
subst_config_alloc_full(const subst_list_type *define_list);
extern "C" void subst_config_free(subst_config_type *subst_config);

extern "C" subst_list_type *
subst_config_get_subst_list(subst_config_type *subst_type);

void subst_config_add_internal_subst_kw(subst_config_type *, const char *,
                                        const char *, const char *);
void subst_config_add_subst_kw(subst_config_type *subst_config, const char *key,
                               const char *value);

#endif
