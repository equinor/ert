#pragma once
#include <string>
#include <sys/wait.h>
#include <vector>
pid_t spawn(const char *executable, int argc, const char **argv,
            const char *stdout_file, const char *stderr_file);
pid_t spawn(const char *executable, const std::vector<std::string> &args,
            const char *stdout_file, const char *stderr_file);
pid_t spawn(char *const argv[], const char *stdout_file,
            const char *stderr_file);
int spawn_blocking(const char *executable, int argc, const char **argv,
                   const char *stdout_file, const char *stderr_file);
int spawn_blocking(const char *executable, const std::vector<std::string> &args,
                   const char *stdout_file, const char *stderr_file);
int spawn_blocking(char *const argv[], const char *stdout_file,
                   const char *stderr_file);
