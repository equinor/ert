#pragma once
#include <string>
#include <sys/wait.h>

struct spawn_result {
    int exit_code;
    std::string out;
    std::string err;
};

pid_t spawn(char *const argv[]);
spawn_result spawn_blocking(char *const argv[]);
