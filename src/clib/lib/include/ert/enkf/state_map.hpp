#pragma once

#include <filesystem>
#include <mutex>
#include <vector>

#include <ert/enkf/enkf_types.hpp>

class StateMap {
    std::vector<int> m_state;
    mutable std::mutex m_mutex;

public:
    StateMap() = default;
    StateMap(const std::filesystem::path &filename);
    StateMap(const StateMap &other);

    int size() const;

    /**
     * StateMap objects are equal iff. their internal state array is equal
     *
     * @return True if the internal arrays are equal, false otherwise
     */
    bool operator==(const StateMap &) const;

    /**
     * Assign a new state at the given index, overriding previously set flags.
     * Resize internal array to contain index if index is out of bounds
     *
     * @param index Index in the internal array
     */
    void set(int index, realisation_state_enum new_state);

    /**
     * Get the state at the given index or STATE_UNDEFINED if index is out of bounds
     *
     * @param index Index in the internal enum vector
     * @returns State at index or STATE_UNDEFINED when oob
     */
    realisation_state_enum get(int index) const;

    /**
     * Change the value at index to new_state iff. the value is currently
     * state_mask. Resizes internal array to contain index if the index is out
     * of bounds
     *
     * @param index Index in the internal state array
     * @param state_mask Flags that we expect the value at position index to be set
     * @param new_state Flag that we want to set at the given position index
     */
    void update_matching(size_t index, int state_mask,
                         realisation_state_enum new_state);

    /**
     * Write data to disk, creating directories as needed
     *
     * @param path Path to file
     */
    void write(const std::filesystem::path &path) const;

    /**
     * Read data from disk
     *
     * @returns true if file was read, false otherwise
     */
    bool read(const std::filesystem::path &path);

    /**
     * Get a bool vector where the elements are true at indices where the
     * elements have the flags of select_mask set
     */
    std::vector<bool> select_matching(int select_mask) const;

    /**
     * Assign state to all indices where the boolean is false
     *
     * @param mask Boolean mask
     * @param state State to assign
     */
    void set_from_inverted_mask(const std::vector<bool> &mask,
                                realisation_state_enum state);

    /**
     * Assign state to all indices where the boolean is true
     *
     * @param mask Boolean mask
     * @param state State to assign
     */
    void set_from_mask(const std::vector<bool> &mask,
                       realisation_state_enum state);

    /**
     * Count the states that have all of flags set in state_mask
     *
     * @param state_mask Set of state flags
     * @return Count
     */
    size_t count_matching(int state_mask) const;

    /**
     * Determine whether it is possible to change from state1 to state2.
     *
     * For example, it isn't permitted to go from STATE_PARENT_FAILURE to
     * STATE_HAS_DATA.
     *
     * @param state1 Start state
     * @param state2 Target state
     * @return True if legal, false otherwise
     */
    static bool is_legal_transition(realisation_state_enum state1,
                                    realisation_state_enum state2);
};
