#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <cppitertools/enumerate.hpp>

#include <ert/util/bool_vector.h>
#include <ert/util/int_vector.h>
#include <ert/util/util.h>

#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/state_map.hpp>
#include <ert/except.hpp>
#include <ert/python.hpp>
#include <ert/res_util/file_utils.hpp>

namespace fs = std::filesystem;

namespace {
void read_libecl_vector(std::istream &s, std::vector<State> &v) {
    std::int32_t length{};
    s.read(reinterpret_cast<char *>(&length), sizeof(length));

    /* default_value is used by libecl's auto-resizeable vector_type to fill in
     * the gaps. We don't do that here, but we still have to read the value. */
    std::int32_t default_value{};
    s.read(reinterpret_cast<char *>(&default_value), sizeof(default_value));

    v.resize(length);
    s.read(reinterpret_cast<char *>(&v[0]), sizeof(v[0]) * v.size());

    /* Validate states */
    State valid_states[]{State::undefined, State::initialized, State::has_data,
                         State::load_failure};
    for (auto &state : v) {
        bool found{};
        for (auto valid_state : valid_states)
            if (state == valid_state)
                found = true;

        if (!found)
            state = State::undefined;
    }
}

void write_libecl_vector(std::ostream &s, const std::vector<State> &v) {
    std::int32_t length = v.size();
    s.write(reinterpret_cast<const char *>(&length), sizeof(length));

    /* default_value is used by libecl's auto-resizeable vector_type to fill in
     * the gaps. We don't do that here, but we still have to write the value. */
    std::int32_t default_value{};
    s.write(reinterpret_cast<const char *>(&default_value),
            sizeof(default_value));

    s.write(reinterpret_cast<const char *>(&v[0]), sizeof(v[0]) * v.size());
}
} // namespace

StateMap::StateMap(const fs::path &filename) { read(filename); }

void StateMap::write(const fs::path &path) const {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec /* Error-code is ignored */);
    std::ofstream stream{path};

    if (!stream.is_open())
        util_abort("%s: failed to open:%s for writing \n", __func__,
                   path.c_str());

    stream.exceptions(stream.failbit);
    write_libecl_vector(stream, *this);
}

bool StateMap::read(const fs::path &filename) {
    std::ifstream stream{filename};
    try {
        stream.exceptions(stream.failbit);
        read_libecl_vector(stream, *this);
        return true;
    } catch (std::ios_base::failure &) {
        std::fill(begin(), end(), State::undefined);
        return false;
    }
}

ERT_CLIB_SUBMODULE("state_map", m) {
    using namespace py::literals;

    py::enum_<State>(m, "State", py::arithmetic{})
        .value("UNDEFINED", State::undefined)
        .value("INITIALIZED", State::initialized)
        .value("HAS_DATA", State::has_data)
        .value("LOAD_FAILURE", State::load_failure)
        .export_values();

    const auto indices_with_data = [](const StateMap &self) {
        std::vector<size_t> indices;
        for (auto [index, state] : iter::enumerate(self)) {
            if (state == State::has_data)
                indices.push_back(index);
        }
        return indices;
    };

    py::class_<StateMap, std::shared_ptr<StateMap>>(m, "StateMap")
        .def(py::self == py::self)
        .def("__len__", [](const StateMap &self) { return self.size(); })
        .def(
            "__iter__",
            [](const StateMap &self) { return py::make_iterator(self); },
            py::keep_alive<0, 1>{})
        .def(
            "__getitem__",
            [](const StateMap &self, size_t index) { return self.at(index); },
            "index"_a)
        .def(
            "__setitem__",
            [](StateMap &self, size_t index, State value) {
                self.at(index) = value;
            },
            "index"_a, "value"_a)
        .def("indices_with_data", indices_with_data);
}
