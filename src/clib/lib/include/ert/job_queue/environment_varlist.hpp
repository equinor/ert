#ifndef ENVIRONMENT_VARLIST_H
#define ENVIRONMENT_VARLIST_H

#include <cstdio>

typedef struct env_varlist_struct env_varlist_type;

extern "C" env_varlist_type *env_varlist_alloc();

extern "C" void env_varlist_update_path(env_varlist_type *list,
                                        const char *path_var,
                                        const char *new_path);
extern "C" void env_varlist_setenv(env_varlist_type *list, const char *var,
                                   const char *value);
extern "C" void env_varlist_free(env_varlist_type *list);

#endif
