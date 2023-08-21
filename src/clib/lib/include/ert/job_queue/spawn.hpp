#pragma once
#include <sys/wait.h>
pid_t spawn(const char *executable, int argc, const char **argv,
            const char *stdout_file, const char *stderr_file);
int spawn_blocking(const char *executable, int argc, const char **argv,
                   const char *stdout_file, const char *stderr_file);
