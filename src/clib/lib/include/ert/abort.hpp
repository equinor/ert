#pragma once

#define CHECK_ALLOC(a)                                                         \
    if (!(a)) {                                                                \
        std::perror("Failed to allocate memory!\n");                           \
        abort();                                                               \
    }
