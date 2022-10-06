#include <filesystem>

#include <stdlib.h>
#include <unistd.h>

#include <ert/util/util.hpp>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    int sleep_time;
    util_sscanf_int(argv[2], &sleep_time);
    sleep(sleep_time);

    char *filename = util_alloc_filename(argv[1], "OK", "status");

    if (fs::exists(argv[1])) {
        FILE *file = util_fopen(filename, "w");
        fprintf(file, "All good");
        fclose(file);
        exit(0);
    } else
        exit(1);
}
