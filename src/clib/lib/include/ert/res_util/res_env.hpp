#ifndef RESENV_H
#define RESENV_H

#include <string>
#include <vector>

#include <stdbool.h>

std::vector<std::string> res_env_alloc_PATH_list();
char *res_env_alloc_PATH_executable(const char *executable);
void res_env_setenv(const char *variable, const char *value);
const char *res_env_interp_setenv(const char *variable, const char *value);
void res_env_unsetenv(const char *variable);
char *res_env_alloc_envvar(const char *value);
char *res_env_isscanf_alloc_envvar(const char *string, int env_index);
const char *res_env_update_path_var(const char *variable, const char *value,
                                    bool append);

#endif // RESLOG_H
