#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>

#include <ert/res_util/file_utils.hpp>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/time_map.hpp>
#include <ert/logging.hpp>
#include <ert/python.hpp>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("enkf");

namespace {
void read_libecl_vector(std::istream &s, std::vector<time_t> &v) {
    std::int32_t length{};
    s.read(reinterpret_cast<char *>(&length), sizeof(length));

    /* default_value is used by libecl's auto-resizeable vector_type to fill in
     * the gaps. We don't do that here, but we still have to read the value. */
    time_t default_value{};
    s.read(reinterpret_cast<char *>(&default_value), sizeof(default_value));

    v.resize(length);
    s.read(reinterpret_cast<char *>(&v[0]), sizeof(v[0]) * v.size());
}

void write_libecl_vector(std::ostream &s, const std::vector<time_t> &v) {
    std::int32_t length = v.size();
    s.write(reinterpret_cast<const char *>(&length), sizeof(length));

    /* default_value is used by libecl's auto-resizeable vector_type to fill in
     * the gaps. We don't do that here, but we still have to write the value. */
    time_t default_value{};
    s.write(reinterpret_cast<const char *>(&default_value),
            sizeof(default_value));

    s.write(reinterpret_cast<const char *>(&v[0]), sizeof(v[0]) * v.size());
}
} // namespace

// int TimeMap::lookup_time_with_tolerance(time_t time,
//                                         int seconds_before_tolerance,
//                                         int seconds_after_tolerance) const {
//     int nearest_index = -1;
//     {
//         if (m_valid_time(time)) {
//             time_t nearest_diff = 999999999999;
//             int current_index = 0;
//             while (true) {
//                 time_t diff = time - get(current_index);
//                 if (diff == 0) {
//                     nearest_index = current_index;
//                     break;
//                 }

//                 if (std::fabs(diff) < nearest_diff) {
//                     bool inside_tolerance = true;
//                     if (seconds_after_tolerance >= 0) {
//                         if (diff >= seconds_after_tolerance)
//                             inside_tolerance = false;
//                     }

//                     if (seconds_before_tolerance >= 0) {
//                         if (diff <= -seconds_before_tolerance)
//                             inside_tolerance = false;
//                     }

//                     if (inside_tolerance) {
//                         nearest_diff = diff;
//                         nearest_index = current_index;
//                     }
//                 }

//                 current_index++;

//                 if (current_index >= m_map.size())
//                     break;
//             }
//         }
//     }
//     return nearest_index;
// }

// std::vector<int> TimeMap::make_index_map(const ecl_sum_type *ecl_sum) const {
//     std::vector<int> index_map;
//     index_map.resize(m_map.size(), -1);

//     int sum_index = ecl_sum_get_first_report_step(ecl_sum);
//     int time_map_index = ecl_sum_get_first_report_step(ecl_sum);
//     for (; time_map_index < m_map.size(); ++time_map_index) {
//         time_t map_time = m_map[time_map_index];
//         if (map_time == DEFAULT_TIME)
//             continue;

//         for (; sum_index <= ecl_sum_get_last_report_step(ecl_sum);
//              ++sum_index) {
//             time_t sum_time = ecl_sum_get_report_time(ecl_sum, sum_index);
//             if (sum_time == map_time)
//                 break;

//             if (sum_time > map_time) {
//                 int day, month, year;
//                 util_set_date_values_utc(map_time, &day, &month, &year);
//                 util_abort("%s: The eclipse summary cases is missing data for "
//                            "date:%4d-%02d-%02d - aborting\n",
//                            __func__, year, month, day);
//             }
//         }

//         if (sum_index > ecl_sum_get_last_report_step(ecl_sum)) {
//             logger->error("Inconsistency in time_map - data will be ignored");
//             break;
//         }

//         index_map[time_map_index] = sum_index;
//     }

//     return index_map;
// }

namespace {

struct mismatch_error {
    size_t index;
    Clock::time_point current_time{};
    Clock::time_point other_time{};
};

bool validate_and_set_data_point(size_t index, SparseTimeArray &time_array,
                                 time_t other_time_t, size_t index,
                                 std::vector<mismatch_error> &errors) {
    auto current_time = time_array[index];
    auto other_time = Clock::from_time_t(other_time_t);

    // The map has not been initialized if current_time == DEFAULT_TIME,
    // if it has been initialized it must be in sync with update_time
    if (current_time && *current_time != other_time) {
        errors.emplace_back(index, *current_time, other_time);
    } else {
        time_array.insert(index, other_time);
    }
}
} // namespace

template <> struct ::fmt::formatter<mismatch_error> {
    template <typename FormatContext>
    auto format(const mismatch_error &err, FormatContext &ctx) const
        -> decltype(ctx.out()) {
        return fmt::format_to(
            ctx.out(),
            "Time mismatch for step: {}, response time: {}, reference case: {}",
            err.index, err.current_time, err.other_time);
    }
};

std::string time_array::validate_or_extend(SparseTimeArray &time_array,
                                           const ecl_sum_type *ecl_sum) {
    std::vector<mismatch_error> errors;

    int begin = ecl_sum_get_first_report_step(ecl_sum);
    int end = ecl_sum_get_last_report_step(ecl_sum);

    /* The range begin, end is inclusive (hence 'end + 1'), and we add an
     * additional element for the start time */
    time_array.resize(end - begin + 2);

    /* First point is handled separately from the rest */
    validate_data_point(0, time_array, ecl_sum_get_start_time(ecl_sum), errors);

    for (int i = begin; i <= end; ++i) {
        if (ecl_sum_has_report_step(ecl_sum, step))
            continue;

        validate_data_point(step, time_array,
                            ecl_sum_get_report_time(ecl_sum, step), errors);
    }

    if (errors.empty())
        return {};

    return fmt::format("{} inconsistencies in time_map, first: {}, last: {}",
                       errors.size(), errors.front(), errors.back());
}

SparseTimeArray time_array::load(const fs::path &path) {
    SparseTimeArray array;
    std::ifstream stream{path, std::ios_base::binary};
    stream.exceptions(stream.failbit);
    read_libecl_vector(stream, array);
    return array;
}

void time_array::write(const fs::path &path, const SparseTimeArray &array) {
    std::ofstream stream{path, std::ios_base::binary};
    stream.exceptions(stream.failbit);
    write_libecl_vector(stream, array);
}

std::optional<SparseTimeArray> time_array::parse_file(const fs::path &path) {
    std::ifstream stream{path};
    stream.exceptions(stream.failbit);

    SparseTimeArray array;

    while (!stream.eof()) {
        std::string date_string;
        stream >> date_string;

        time_t date;
        /* Try read ISO8601 date (YYYY-MM-DD) */
        if (!util_sscanf_isodate(date_string.c_str(), &date)) {
            logger->warning("** Deprecation warning: The date format as in "
                            "\'{}\' is deprecated, and its support will be "
                            "removed in a future release. Please use ISO "
                            "date format YYYY-MM-DD.\n",
                            date_string);

            /* Fall back to deprecated Norwegian date format (DD.MM.YYYY) */
            if (!util_sscanf_date_utc(date_string.c_str(), &date)) {
                logger->error("** ERROR: The string \'{}\' was not correctly "
                              "parsed as a date. "
                              "Please use ISO date format YYYY-MM-DD.\n",
                              date_string);
                return false;
            }
        }

        if (date > last_date) {
            array.emplace_back(Clock::from_time_t(date));
            last_date = date;
        } else {
            logger->error("** ERROR: The dates in {} must be in "
                          "strictly increasing order\n",
                          filename);
            return false;
        }
    }

    return array;
}

bool time_array::compare_with_eclipse(const SparseTimeArray &time_array,
                                      const ecl_sum_type *refcase) {
    auto first_step = ecl_sum_get_first_report_step(refcase);
    auto last_step = ecl_sum_get_last_report_step(refcase);

    size_t size = last_step - first_step + 2;

    if (time_array.size() < size) {
        /* Disallow refcases longer than time_array */
        return false;
    }

    for (auto i = first_step; i < last_step; ++i) {
        auto current_time = time_array[i];
        auto refcase_time =
            Clock::from_time_t(ecl_sum_get_report_time(refcase, step));

        if (current_time != refcase_time)
            return false;
    }

    return true;
}
