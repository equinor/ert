#ifndef RESENV_H
#define RESENV_H

#include <string>
#include <vector>

#include <stdbool.h>

char *res_env_alloc_PATH_executable(const char *executable);
void res_env_setenv(const char *variable, const char *value);
void res_env_unsetenv(const char *variable);
char *res_env_isscanf_alloc_envvar(const char *string, int env_index);

#endif // RESLOG_H
