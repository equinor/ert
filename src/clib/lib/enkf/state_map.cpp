#include <algorithm>
#include <cstdint>
#include <filesystem>

#include <fstream>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <stdlib.h>

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
void read_libecl_vector(std::istream &s, std::vector<int> &v) {
    std::int32_t length{};
    s.read(reinterpret_cast<char *>(&length), sizeof(length));

    /* default_value is used by libecl's auto-resizeable vector_type to fill in
     * the gaps. We don't do that here, but we still have to read the value. */
    std::int32_t default_value{};
    s.read(reinterpret_cast<char *>(&default_value), sizeof(default_value));

    v.resize(length);
    s.read(reinterpret_cast<char *>(&v[0]), sizeof(v[0]) * v.size());
}

void write_libecl_vector(std::ostream &s, const std::vector<int> &v) {
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

int StateMap::size() const {
    std::lock_guard guard{m_mutex};
    return m_state.size();
}

bool StateMap::operator==(const StateMap &other) const {
    std::scoped_lock lock{m_mutex, other.m_mutex};
    return m_state == other.m_state;
}

realisation_state_enum StateMap::get(int index) const {
    std::lock_guard guard{m_mutex};
    if (index < m_state.size())
        return static_cast<realisation_state_enum>(m_state.at(index));
    return realisation_state_enum::STATE_UNDEFINED;
}

bool StateMap::is_legal_transition(realisation_state_enum state1,
                                   realisation_state_enum state2) {
    int target_mask = 0;

    if (state1 == STATE_UNDEFINED)
        target_mask = STATE_INITIALIZED | STATE_PARENT_FAILURE;
    else if (state1 == STATE_INITIALIZED)
        target_mask = STATE_LOAD_FAILURE | STATE_HAS_DATA | STATE_INITIALIZED |
                      STATE_PARENT_FAILURE;
    else if (state1 == STATE_HAS_DATA)
        target_mask = STATE_INITIALIZED | STATE_LOAD_FAILURE | STATE_HAS_DATA |
                      STATE_PARENT_FAILURE;
    else if (state1 == STATE_LOAD_FAILURE)
        target_mask = STATE_HAS_DATA | STATE_INITIALIZED | STATE_LOAD_FAILURE;
    else if (state1 == STATE_PARENT_FAILURE)
        target_mask = STATE_INITIALIZED | STATE_PARENT_FAILURE;

    if (state2 & target_mask)
        return true;
    else
        return false;
}

void StateMap::set(int index, realisation_state_enum new_state) {
    std::lock_guard lock{m_mutex};

    auto current_state = static_cast<realisation_state_enum>(m_state.at(index));

    if (is_legal_transition(current_state, new_state))
        m_state[index] = new_state;
    else
        util_abort(
            "%s: illegal state transition for realisation:%d %d -> %d \n",
            __func__, index, current_state, new_state);
}

void StateMap::update_matching(size_t index, int state_mask,
                               realisation_state_enum new_state) {
    realisation_state_enum current_state = get(index);
    if (current_state & state_mask)
        set(index, new_state);
}

void StateMap::write(const fs::path &path) const {
    std::lock_guard lock{m_mutex};

    std::error_code ec;
    fs::create_directories(path.parent_path(), ec /* Error-code is ignored */);
    std::ofstream stream{path};

    if (!stream.is_open())
        util_abort("%s: failed to open:%s for writing \n", __func__,
                   path.c_str());

    stream.exceptions(stream.failbit);
    write_libecl_vector(stream, m_state);
}

bool StateMap::read(const fs::path &filename) {
    std::lock_guard lock{m_mutex};
    std::ifstream stream{filename};
    try {
        stream.exceptions(stream.failbit);
        read_libecl_vector(stream, m_state);
        return true;
    } catch (std::ios_base::failure &) {
        std::fill(m_state.begin(), m_state.end(), STATE_UNDEFINED);
        return false;
    }
}

std::vector<bool> StateMap::select_matching(int select_mask) const {
    std::lock_guard lock{m_mutex};
    std::vector<bool> select_target(m_state.size(), false);
    for (size_t i{}; i < m_state.size(); ++i) {
        auto state_value = m_state[i];
        if (state_value & select_mask)
            select_target[i] = true;
    }
    return select_target;
}

ERT_CLIB_SUBMODULE("state_map", m) {
    using namespace py::literals;

    py::enum_<realisation_state_enum>(m, "RealizationStateEnum",
                                      py::arithmetic{})
        .value("STATE_UNDEFINED", STATE_UNDEFINED)
        .value("STATE_INITIALIZED", STATE_INITIALIZED)
        .value("STATE_HAS_DATA", STATE_HAS_DATA)
        .value("STATE_LOAD_FAILURE", STATE_LOAD_FAILURE)
        .value("STATE_PARENT_FAILURE", STATE_PARENT_FAILURE)
        .export_values();

    py::class_<StateMap, std::shared_ptr<StateMap>>(m, "StateMap")
        .def_static("isLegalTransition", &StateMap::is_legal_transition)
        .def(py::self == py::self)
        .def("__len__", &StateMap::size)
        .def("_get", &StateMap::get, "index"_a)
        .def("_set", &StateMap::set, "index"_a, "new_state"_a)
        .def("__setitem__", &StateMap::set, "index"_a, "value"_a)
        .def("selectMatching", [](const StateMap &x,
                                  int mask) { return x.select_matching(mask); })
        .def("update_matching", [](StateMap &x, size_t index, int state_mask,
                                   realisation_state_enum new_state) {
            return x.update_matching(index, state_mask, new_state);
        });
}
