#include <filesystem>
#include <fstream>

#include <ert/enkf/gen_common.hpp>
#include <ert/except.hpp>

namespace fs = std::filesystem;

namespace {
std::vector<double> parse_text(const fs::path &path) {
    std::ifstream stream{path};
    stream.imbue(std::locale::classic());

    std::vector<double> data;
    for (;;) {
        double value;
        if (!(stream >> value))
            break;
        data.emplace_back(value);
        stream >> std::ws;
    }
    if (!stream.eof())
        throw exc::runtime_error{
            "Could not parse contents of {} as a sequence of numbers", path};

    return data;
}
} // namespace

std::vector<double>
gen_common_fload_alloc(const fs::path &path,
                       gen_data_file_format_type load_format) {
    if (load_format == ASCII) {
        return parse_text(path);
    } else {
        throw exc::runtime_error{"Invalid GEN data format: {}", load_format};
    }
}
