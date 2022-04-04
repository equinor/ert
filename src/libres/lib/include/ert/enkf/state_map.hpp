#pragma once

#include <filesystem>
#include <mutex>
#include <vector>

#include <ert/enkf/enkf_types.hpp>

class StateMap {
    std::vector<int> m_state;
    mutable std::mutex m_mutex;
    bool m_read_only{};

    void m_assert_writable() const;

public:
    StateMap() = default;
    StateMap(const std::filesystem::path &filename, bool read_only = true);
    StateMap(const StateMap &other);

    int size() const;
    bool operator==(const StateMap &) const;

    bool is_readonly() const { return m_read_only; };

    void set(int index, realisation_state_enum new_state);
    realisation_state_enum get(int index) const;
    void update_matching(size_t index, int state_mask,
                         realisation_state_enum new_state);
    void update_undefined(size_t index, realisation_state_enum new_state) {
        update_matching(index, STATE_UNDEFINED, new_state);
    }

    void write(const std::filesystem::path &path) const;
    bool read(const std::filesystem::path &path);
    bool read(const std::filesystem::path &path, bool read_only);

    std::vector<bool> select_matching(int select_mask, bool select) const;
    void set_from_inverted_mask(const std::vector<bool> &mask,
                                realisation_state_enum state);
    void set_from_mask(const std::vector<bool> &mask,
                       realisation_state_enum state);
    size_t count_matching(int mask) const;

    static bool is_legal_transition(realisation_state_enum state1,
                                    realisation_state_enum state2);
};
