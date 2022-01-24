#pragma once

#include <unordered_set>

namespace enkf {
enum class ActiveMode { all_active, inactive, partly_active };

class ActiveList {
    ActiveMode m_mode = ActiveMode::all_active;
    std::unordered_set<int> m_indices;

public:
    ActiveList() = default;
    ActiveList(const ActiveList&) = default;
    ActiveList(ActiveList&&) = default;

    ActiveList& operator=(const ActiveList&) = default;
    ActiveList& operator=(ActiveList&&) = default;

    bool operator[](int index) const;
    bool operator==(const ActiveList& other) const;

    ActiveMode mode() const { return m_mode; }
    int active_size(int active_size) const;

    void add_index(int new_index);

    auto begin() const { return m_indices.begin(); }
    auto end() const { return m_indices.end(); }
};
}

// void active_list_summary_fprintf(const enkf::ActiveList &active_list,
//                                  const char *dataset_key, const char *key,
//                                  FILE *stream);
