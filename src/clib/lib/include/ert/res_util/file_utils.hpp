#ifndef ERT_FILE_UTILS_H
#define ERT_FILE_UTILS_H

#include <filesystem>

namespace fs = std::filesystem;

/**
   Open file-stream to "/some/path/to/file.txt" without first ensuring
   that "/some/path/to" exists.
*/
FILE *mkdir_fopen(fs::path, const char *);

#endif
