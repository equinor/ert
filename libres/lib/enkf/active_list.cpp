#include <ert/enkf/active_list.hpp>
#include <ert/python.hpp>

void enkf::ActiveList::add_index(int new_index) {
    m_indices.emplace(new_index);
    m_mode = ActiveMode::partly_active;
}

/*
   When mode == PARTLY_ACTIVE the active_list instance knows the size
   of the active set; if the mode is INACTIVE 0 will be returned and
   if the mode is ALL_ACTIVE the input parameter @total_size will be
   passed back to calling scope.
*/

int enkf::ActiveList::active_size(int total_size) const {
    switch (m_mode) {
    case ActiveMode::partly_active:
        return m_indices.size();
    case ActiveMode::inactive:
        return 0;
    case ActiveMode::all_active:
        return total_size;
    }
    __builtin_unreachable();
}

bool enkf::ActiveList::operator[](int index) const {
    if (m_mode == ActiveMode::all_active)
        return true;
    if (m_mode == ActiveMode::inactive)
        return false;
    return m_indices.count(index) > 0;
}

bool enkf::ActiveList::operator==(const enkf::ActiveList& other) const {
    if (m_mode != other.m_mode)
        return false;

    if (m_mode == ActiveMode::partly_active)
        return m_indices == other.m_indices;

    return true;
}

namespace {
    std::vector<int> to_list(const enkf::ActiveList& active_list) {
        std::vector<int> list;
        for (auto x : active_list)
            list.push_back(x);
        return list;
    }
}

RES_LIB_SUBMODULE("active_list", m) {
    using namespace py::literals;

    py::enum_<enkf::ActiveMode>(m, "ActiveMode")
        .value("ALL_ACTIVE", enkf::ActiveMode::all_active)
        .value("INACTIVE", enkf::ActiveMode::inactive)
        .value("PARTY_ACTIVE", enkf::ActiveMode::partly_active);

    py::class_<enkf::ActiveList, std::shared_ptr<enkf::ActiveList>>(m, "ActiveList")
        .def(py::init<>())
        .def("getMode", &enkf::ActiveList::mode)
        .def("get_active_index_list", &to_list)
        .def("addActiveIndex", &enkf::ActiveList::add_index, "index"_a)
        .def("getActiveSize", &enkf::ActiveList::active_size, "default_value"_a);
}
