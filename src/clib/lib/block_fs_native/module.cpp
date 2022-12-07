#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cppitertools/enumerate.hpp>
#include <fmt/format.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <zlib.h>

using namespace std::string_view_literals;
namespace py = pybind11;
namespace fs = std::filesystem;

enum struct Kind {
    field = 104,
    gen_kw = 107,
    summary = 110,
    gen_data = 113,
    surface = 114,
    ext_param = 116,
};
constexpr size_t NUM_KINDS = 6;

auto kind2index(Kind kind) -> size_t {
    switch (kind) {
    case Kind::field:
        return 0;
    case Kind::gen_kw:
        return 1;
    case Kind::summary:
        return 2;
    case Kind::gen_data:
        return 3;
    case Kind::surface:
        return 4;
    case Kind::ext_param:
        return 5;
    default:
        throw py::value_error{"Unknown block kind"};
    }
}

struct Block {
    Kind kind;
    std::string name;
    int report_step;
    int realization_index;
    ssize_t pos;
    size_t len;
    size_t count;
};

template <typename T>
[[nodiscard]] auto read(std::istream &s, ssize_t *size = nullptr) -> T {
    T data;
    s.read(reinterpret_cast<char *>(&data), sizeof(data));
    if (size)
        *size -= sizeof(T);
    return data;
}

template <typename T> void skip(std::istream &s, ssize_t *size = nullptr) {
    s.seekg(sizeof(T), s.cur);
    if (size)
        *size -= sizeof(T);
}

// Sanity-check standard types
static_assert(sizeof(double) == 8);

const auto read_u32 = read<uint32_t>;
const auto read_i32 = read<int32_t>;
const auto skip_i32 = skip<int32_t>;
const auto skip_i64 = skip<int64_t>;
const auto skip_f64 = skip<double>;

/**
 * BlockFs does not have an index in the main file. Its contents must be found
 * by finding a sequence of 4 0x55 bytes. This is the NODE_IN_USE marker.
 */
void seek_until_marker(std::istream &s) {
    size_t count{};

    while (count < 4) {
        count = (s.get() == 0x55) ? count + 1 : 0;
    }
}

auto parse_name(const std::string &name, Kind kind)
    -> std::tuple<std::string, int, int> {
    auto index = name.rfind('.');
    if (index == name.npos) {
        throw py::value_error(
            fmt::format("Key '{}' has no realization index", name));
    }
    if (kind == Kind::summary) {
        // SUMMARY blocks do not contain report_step, so we are done
        return {name.substr(0, index), 0, std::stoi(name.substr(index + 1))};
    }

    auto index_ = name.rfind('.', index - 1);
    if (index_ == name.npos) {
        throw py::value_error(fmt::format("Key '{}' has no report step", name));
    }

    return {name.substr(0, index_), std::stoi(name.substr(index_ + 1, index)),
            std::stoi(name.substr(index + 1))};
}

/**
 *
 */
class DataFile {
    /** Stream to '.data_0' file */
    std::ifstream m_stream;
    /** Stream mutex */
    mutable std::mutex m_mutex;
    /** List of blocks for each kind */
    std::array<std::vector<Block>, NUM_KINDS> m_blocks{};
    /** Set of realisation indices */
    std::unordered_set<size_t> m_realizations{};

    void _load_vector(const Block &block, double *vector, size_t count) {
        std::lock_guard _mutex_guard{m_mutex};
        py::gil_scoped_release _release_gil;

        m_stream.clear();
        m_stream.seekg(block.pos, std::ios::beg);
        m_stream.read(reinterpret_cast<char *>(vector),
                      sizeof(*vector) * block.count);
    }

    template <typename T>
    void _load_vector_compressed(const Block &block, T *vector, size_t count) {
        std::lock_guard _mutex_guard{m_mutex};
        py::gil_scoped_release _release_gil;

        m_stream.clear();
        m_stream.seekg(block.pos, std::ios::beg);

        std::string compressed;
        compressed.resize(block.len);
        m_stream.read(compressed.data(), block.len);

        uLongf destlen = sizeof(T) * count;
        uncompress(reinterpret_cast<Bytef *>(vector), &destlen,
                   reinterpret_cast<Bytef *>(compressed.data()),
                   compressed.size());
    }

public:
    DataFile(const fs::path &path) : m_stream(path) { this->build_index(); }

    void build_index() {
        py::gil_scoped_release _release_gil;

        auto exceptions = m_stream.exceptions();
        m_stream.exceptions(std::ifstream::failbit);
        try {
            while (!m_stream.eof()) {
                seek_until_marker(m_stream);

                size_t name_length = read_u32(m_stream, nullptr);
                std::string name;
                name.resize(name_length);
                m_stream.read(name.data(), name.size());
                skip<char>(m_stream); // NULL terminator

                // Skip node_size
                skip_i32(m_stream, nullptr);
                ssize_t data_size = read_i32(m_stream, nullptr);

                skip_i64(m_stream, &data_size);
                size_t count = 0;

                auto kind = static_cast<Kind>(read_i32(m_stream, &data_size));
                switch (kind) {
                case Kind::summary:
                    // Read count
                    count = read_u32(m_stream, &data_size);

                    // Skip default value
                    skip_f64(m_stream, &data_size);
                    break;

                case Kind::gen_data:
                    // Read count
                    count = read_u32(m_stream, &data_size);

                    // Skip report_step
                    skip_i32(m_stream, &data_size);
                    break;

                case Kind::surface:
                case Kind::gen_kw:
                case Kind::ext_param:
                    // The count is given in the config and not available in the
                    // data file, but we can make an informed guess by looking
                    // at the size of the whole data section
                    count = data_size / sizeof(double);
                    break;

                case Kind::field:
                    // The count is given in the config
                    break;

                default:
                    continue;
                }

                auto [name_, report_step, realization_index] =
                    parse_name(name, kind);
                m_blocks[kind2index(kind)].push_back({
                    .kind = kind,
                    .name = name_,
                    .report_step = report_step,
                    .realization_index = realization_index,
                    .pos = m_stream.tellg(),
                    .len = static_cast<size_t>(data_size),
                    .count = count,
                });
                m_realizations.insert(realization_index);
            }
        } catch (std::ios_base::failure &) {
            /* ignore */
        }
        m_stream.exceptions(exceptions);
    }

    /**
     * FIELD blocks are 32-bit single-precision floats rather than 64-bit double-precision floats, so they are handled separately.
     */
    auto load_field(const Block &block, size_t count_hint)
        -> py::array_t<float, 1> {
        std::array<size_t, 1> shape{{count_hint}};
        py::array_t<float, 1> array(shape);
        this->_load_vector_compressed(block, array.mutable_data(), count_hint);
        return array;
    }

    auto load(const Block &block, std::optional<size_t> count_hint)
        -> py::array_t<double, 1> {
        size_t count{};
        switch (block.kind) {
        case Kind::gen_kw:
        case Kind::surface:
        case Kind::ext_param:
            if (count_hint.has_value() && *count_hint != block.count) {
                throw py::value_error(fmt::format(
                    "On-disk vector has {} elements, but ERT config expects {}",
                    block.count, *count_hint));
            }
            [[fallthrough]];

        case Kind::summary:
        case Kind::gen_data:
            count = block.count;
            break;

        default:
            throw py::type_error("Unknown block kind");
        }

        std::array<size_t, 1> shape{{count}};
        py::array_t<double, 1> array{shape};
        if (block.kind == Kind::gen_data) {
            this->_load_vector_compressed(block, array.mutable_data(), count);
        } else {
            this->_load_vector(block, array.mutable_data(), count);
        }
        return array;
    }

    auto blocks(Kind kind) const -> const std::vector<Block> & {
        return m_blocks[kind2index(kind)];
    }

    auto realizations() const -> const std::unordered_set<size_t> & {
        return m_realizations;
    }
};

PYBIND11_MODULE(_block_fs_native, m) {
    using namespace py::literals;

    py::enum_<Kind>(m, "Kind", py::arithmetic{})
        .value("FIELD", Kind::field)
        .value("GEN_KW", Kind::gen_kw)
        .value("SUMMARY", Kind::summary)
        .value("GEN_DATA", Kind::gen_data)
        .value("SURFACE", Kind::surface)
        .value("EXT_PARAM", Kind::ext_param);

    py::class_<Block>(m, "Block")
        .def("__str__", [](const Block &self) { return self.name; })
        .def("__repr__",
             [](const Block &self) {
                 return fmt::format("Block(name=\"{}\", realization_index={})",
                                    self.name, self.realization_index);
             })
        .def_readonly("kind", &Block::kind)
        .def_readonly("name", &Block::name)
        .def_readonly("report_step", &Block::report_step)
        .def_readonly("realization_index", &Block::realization_index);

    py::class_<DataFile>(m, "DataFile")
        .def(py::init<const fs::path &>(), "path"_a)
        .def("load_field", &DataFile::load_field, "block"_a, "count_hint"_a)
        .def("load", &DataFile::load, "block"_a, "count_hint"_a)
        .def(
            "blocks",
            [](const DataFile &self, Kind kind) {
                return py::make_iterator(self.blocks(kind));
            },
            py::keep_alive<0, 1>())
        .def_property_readonly("realizations", &DataFile::realizations);

    m.def("parse_name", &parse_name, "name"_a, "is_summary"_a);
}
