#include <ert/enkf/active_list.hpp>

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
}

bool enkf::ActiveList::operator[](int index) const {
    if (m_mode == ActiveMode::all_active)
        return true;
    if (m_mode == ActiveMode::inactive)
        return false;
    return m_indices.count(index) > 0;
}

// void active_list_summary_fprintf(const enkf::ActiveList &active_list,
//                                  const char *dataset_key, const char *key,
//                                  FILE *stream) {
//     int number_of_active = int_vector_size(active_list->index_list);
//     if (active_list->mode == ALL_ACTIVE) {
//         fprintf(stream, "NUMBER OF ACTIVE:%d,STATUS:%s,", number_of_active,
//                 "ALL_ACTIVE");
//     } else if (active_list->mode == PARTLY_ACTIVE) {
//         fprintf(stream, "NUMBER OF ACTIVE:%d,STATUS:%s,", number_of_active,
//                 "PARTLY_ACTIVE");
//     } else
//         fprintf(stream, "NUMBER OF ACTIVE:%d,STATUS:%s,", number_of_active,
//                 "INACTIVE");
// }

bool enkf::ActiveList::operator==(const enkf::ActiveList& other) const {
    if (m_mode != other.m_mode)
        return false;

    if (m_mode == ActiveMode::partly_active)
        return m_indices == other.m_indices;

    return true;
}
