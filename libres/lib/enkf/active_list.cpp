#include <ert/enkf/active_list.hpp>

void enkf::ActiveList::add_index(int new_index) {
    m_indices.emplace(new_index);
    m_mode = ActiveMode::partly_active;
}

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
