#pragma once

#include <chrono>
#include <ctime>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <ert/except.hpp>

typedef struct ecl_sum_struct ecl_sum_type;

/**
 */
using Clock = std::chrono::system_clock;

/**
 * A vector or milliseconds. These are meant to be directly convertable to
 * numpy's "datetime[ms]" dtype. Both represent milliseconds in POSIX time.
 */
using TimeArray = std::vector<Clock::time_point>;

/**
 * A an array of time points that is allowed to have gaps.
 *
 * Previously the time point '-1' was used to mean non-existent time. However,
 * this is a valid datetime representing 1 time unit before midnight of
 * 1970-01-01 UTC. To allow this, we use a separate
 *
 * @note Not optimised for sparsity, it just allows it
 */
class SparseTimeArray {
    std::vector<bool> m_exist;
    std::vector<Clock::time_point> m_data;

    void m_assert_index(size_t i) const {
        if (i < m_exist.size()) {
            throw exc::out_of_range("{} out of range in vector of size {}",
                                    __PRETTY_FUNCTION__, i, size());
        }
    }

public:
    using value_type = std::optional<std::reference_wrapper<Clock::time_point>>;
    using const_value_type = std::optional<Clock::time_point>;

    void clear() {
        m_exist.clear();
        m_data.clear();
    }

    void resize(size_t new_size) {
        m_exist.resize(new_size);
        m_data.resize(new_size);
    }

    auto size() const -> decltype(m_exist.size()) { return m_exist.size(); }

    auto at(size_t i) -> value_type {
        m_assert_index(i);
        return m_exist[i] ? std::optional{std::ref(m_data[i])} : std::nullopt;
    }

    auto at(size_t i) const -> const_value_type {
        m_assert_index(i);
        return m_exist[i] ? std::optional{m_data[i]} : std::nullopt;
    }

#ifdef NDEBUG
    auto operator[](size_t i) -> decltype(this->at(i)) {
        return m_exist[i] ? std::optional{std::ref(m_data[i])} : std::nullopt;
    }

    auto operator[](size_t i) const -> decltype(this->at(i)) {
        return m_exist[i] ? std::optional{m_data[i]} : std::nullopt;
    }
#else
    /* Debug versions of the getter functions are wrappers around the exception-throwing 'at' */
    auto operator[](size_t i) -> decltype(this->at(i)) { return at(i); }
    auto operator[](size_t i) const -> decltype(this->at(i)) { return at(i); }
#endif

    void insert(size_t index, Clock::time_point value) {
#ifndef NDEBUG
        m_assert_index(index);
#endif

        m_exist[index] = true;
        m_data[index] = value;
    }

    void emplace_back(Clock::time_point value) {
        m_exist.emplace_back(true);
        m_data.emplace_back(value);
    }
};

/**
 * Standalone functions for dealing with TimeArray
 */
namespace time_array {
/**
 * Validate that the
 */
std::string validate_or_extend(TimeArray &time_array,
                               const ecl_sum_type *ecl_sum);

/**
 * Count the number of days from the start of the
 */
size_t count_days_from_beginning(const std::vector<time_t> &, size_t index);

/**
 * Load time map from a binary file
 *
 * The format is that of a libecl time_t vector. The array is guaranteed to be
 * sorted.
 *
 * @return Array of loaded and sorted time_t
 */
TimeArray load(const std::filesystem::path &);

/**
 * Write time map into a binary file
 *
 * The format is that of a libecl time_t vector.
 */
void write(const std::filesystem::path &, const SparseTimeArray &);

/**
 * Load time map from a text file
 *
 * Each entry in the file is in the form "DD.MM.YYYY", where '.' represents any
 * character, separated by whitespace (including newline).
 *
 * This file can only be written by the user hence there is no write version of
 * this function.
 *
 * @return Array of parsed and sorted time_t
 */
std::optional<TimeArray> parse_file(const std::filesystem::path &);

/**
 * Load time map from an ECLIPSE summary
 */
TimeArray parse_eclipse(const ecl_sum_type *summary);

/**
   The refcase will only be attached if it is consistent with the
   current time map; we will accept attaching a refcase which is
   shorter than the current case.
*/
bool compare_with_eclipse(const SparseTimeArray &time_array,
                          const ecl_sum_type *refcase);
} // namespace time_array
