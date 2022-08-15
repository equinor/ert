#include <filesystem>

#include <ert/res_util/file_utils.hpp>

namespace fs = std::filesystem;

FILE *mkdir_fopen(fs::path file_path, const char *mode) {
    auto file_name = file_path.filename();
    auto directory = file_path.remove_filename();
    if (!directory.empty())
        fs::create_directories(directory);
    auto full_path = directory / file_name;
    FILE *stream = fopen(full_path.c_str(), mode);
    return stream;
}
