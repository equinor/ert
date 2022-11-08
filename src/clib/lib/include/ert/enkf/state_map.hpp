#pragma once

#include <filesystem>
#include <mutex>
#include <vector>

enum struct State : std::int32_t {
    undefined = 1,
    initialized = 2,
    has_data = 4,
    load_failure = 8,
};

class StateMap : public std::vector<State> {
    using Super = std::vector<State>;

public:
    StateMap() = delete;
    StateMap(size_t ensemble_size) : Super(ensemble_size, State::undefined) {}
    StateMap(const std::filesystem::path &filename);

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
    std::vector<bool> select_matching(State state) const;
};
