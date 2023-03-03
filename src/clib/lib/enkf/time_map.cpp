#include "ert/except.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/time_map.hpp>
#include <ert/logging.hpp>
#include <ert/python.hpp>
#include <stdexcept>
#include <string>
#include <system_error>

#include <pybind11/chrono.h>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("enkf");

namespace {
void read_libecl_vector(std::istream &s, std::vector<time_t> &v) {
    std::int32_t length{};
    s.read(reinterpret_cast<char *>(&length), sizeof(length));

    /* default_value is used by libecl's auto-resizeable vector_type to fill in
     * the gaps. We don't do that here, but we still have to read the value. */
    std::time_t default_value{};
    s.read(reinterpret_cast<char *>(&default_value), sizeof(default_value));

    v.resize(length);
    s.read(reinterpret_cast<char *>(&v[0]), sizeof(v[0]) * v.size());
}

void write_libecl_vector(std::ostream &s, const std::vector<time_t> &v) {
    std::int32_t length = v.size();
    s.write(reinterpret_cast<const char *>(&length), sizeof(length));

    /* default_value is used by libecl's auto-resizeable vector_type to fill in
     * the gaps. We don't do that here, but we still have to write the value. */
    std::time_t default_value{};
    s.write(reinterpret_cast<const char *>(&default_value),
            sizeof(default_value));

    s.write(reinterpret_cast<const char *>(&v[0]), sizeof(v[0]) * v.size());
}
} // namespace

/**
   The refcase will only be attached if it is consistent with the
   current time map; we will accept attaching a refcase which is
   shorter than the current case.
*/
bool TimeMap::attach_refcase(const ecl_sum_type *refcase) {
    size_t ecl_sum_size = ecl_sum_get_last_report_step(refcase) + 1;
    auto max_step = std::min(size(), ecl_sum_size);

    /* Skip the first step as it's "fraught with uncertainty" and not really
     * valid. */
    for (size_t step = 1; step < max_step; ++step) {
        if ((*this)[step] != ecl_sum_get_report_time(refcase, step)) {
            m_refcase = nullptr;
            return false;
        }
    }
    m_refcase = refcase;
    return true;
}

void TimeMap::read_text(const std::filesystem::path &path) {
    std::ifstream s{path};

    clear();
    size_t txt{};
    while (!s.eof()) {
        std::string token;
        s >> token;
        if (s.fail())
            break;

        txt++;
        s >> std::ws;

        /* Try as ISO-8601 date */
        time_t date;
        if (!util_sscanf_isodate(token.c_str(), &date)) {
            logger->warning("The date format as in '{}' is deprecated, and its "
                            "support will be removed in a future release. "
                            "Please use ISO date format YYYY-MM-DD",
                            token);

            /* Fall back to DD/MM/YYYY date format */
            if (!util_sscanf_date_utc(token.c_str(), &date)) {
                throw exc::runtime_error{
                    "The date '{}' could not be parsed. Please use "
                    "ISO date format YYYY-MM-DD.",
                    token};
            }
        }

        if (!empty() && back() >= date)
            throw exc::runtime_error{"Inconsistent time map"};
        push_back(date);
    }
}

/**
   @brief Checks if time map can be updated and sets value if valid.
   Must hold the write lock. When a refcase is supplied we gurantee
   that all values written into the map agree with the refcase
   values. However the time map is not preinitialized with the refcase
   values.

   @returns status, empty string if success, error message otherwise
*/
std::optional<Inconsistency> TimeMap::m_update(size_t step,
                                               time_t update_time) {
    auto &current_time = (*this)[step];
    // The map has not been initialized if current_time == DEFAULT_TIME,
    // if it has been initialized it must be in sync with update_time
    if ((current_time != DEFAULT_TIME) && (current_time != update_time))
        return Inconsistency{
            .step = step, .expected = update_time, .actual = current_time};

    // We could be in a state where map->map is not initialized, however there
    // is a refcase, in that case, make sure the refcase is in sync.
    if (m_refcase && step <= ecl_sum_get_last_report_step(m_refcase)) {
        time_t ref_time = ecl_sum_get_report_time(m_refcase, step);

        if (ref_time != update_time)
            return Inconsistency{
                .step = step, .expected = update_time, .actual = ref_time};
    }
    current_time = update_time;
    return std::nullopt;
}

/**
   Observe that the locking is opposite of the function name; i.e.
   the time_map_fwrite() function reads the time_map and takes the
   read lock, whereas the time_map_fread() function takes the write
   lock.
*/
void TimeMap::write_binary(const std::filesystem::path &path) const {
    /* Create directories and ignore any errors */
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);

    std::ofstream s{path, std::ios::binary};
    s.exceptions(s.failbit);
    write_libecl_vector(s, *this);
}

void TimeMap::read_binary(const std::filesystem::path &path) {
    clear();

    std::ifstream s{path, std::ios::binary};
    try {
        s.exceptions(s.failbit);
        read_libecl_vector(s, *this);
    } catch (std::ios::failure &ex) {
        clear();
    }
}

/**
   @brief Checks if summary data and time map are in sync, returns
   an error string, empty string means the time map is in sync with
   the summary data.

   @returns status, empty string if success, error message otherwise
*/
std::string TimeMap::summary_update(const ecl_sum_type *ecl_sum) {
    std::vector<Inconsistency> errors;

    int first_step = ecl_sum_get_first_report_step(ecl_sum);
    int last_step = ecl_sum_get_last_report_step(ecl_sum);
    resize(last_step + 1, DEFAULT_TIME);

    if (auto err = m_update(0, ecl_sum_get_start_time(ecl_sum)); err)
        errors.emplace_back(*err);

    for (int step = first_step; step <= last_step; step++) {
        if (!ecl_sum_has_report_step(ecl_sum, step))
            continue;

        if (auto err = m_update(step, ecl_sum_get_report_time(ecl_sum, step));
            err)
            errors.emplace_back(*err);
    }

    if (!errors.empty()) {
        return fmt::format(
            "{} inconsistencies in time_map, first: {}, last: {}",
            errors.size(), errors.front(), errors.back());
    }

    return "";
}

/**
  This function creates an integer index mapping from the time map
  into the summary case. In general the time <-> report step mapping
  of the summary data should coincide exactly with the one maintained
  in the time_map, however we allow extra timesteps in the summary
  instance. The extra timesteps will be ignored, holes in the summary
  timestep is not allowed - that will lead to a hard crash.

     time map                      Summary
     -------------------------------------------------
     0: 2000-01-01   <-------      0: 2000-01-01

     1: 2000-02-01   <-------      1: 2000-02-01

     2: 2000-03-01   <-\           2: 2000-02-02 (Ignored)
                        \
                         \--       3: 2000-03-01

     3: 2000-04-01   <-------      4: 2000-04-01


     index_map = { 0 , 1 , 3 , 4 }

  Observe that TimeMap::update_summary() must be called prior to
  calling this function, to ensure that the time_map is sufficiently
  long. If timesteps are missing from the summary case we crash hard:


     time map                      Summary
     -------------------------------------------------
     0: 2000-01-01   <-------      0: 2000-01-01

     1: 2000-02-01   <-------      1: 2000-02-01

     2: 2000-03-01                 ## ERROR -> util_abort()

     3: 2000-04-01   <-------      2: 2000-04-01

*/
std::vector<int> TimeMap::indices(const ecl_sum_type *ecl_sum) const {
    if (empty())
        return {};

    std::vector<int> indices{-1};

    int sum_index = ecl_sum_get_first_report_step(ecl_sum);
    int time_map_index = ecl_sum_get_first_report_step(ecl_sum);

    indices.resize(time_map_index, -1);
    for (; time_map_index < size(); ++time_map_index) {
        time_t map_time = (*this)[time_map_index];
        if (map_time == DEFAULT_TIME) {
            indices.push_back(-1);
            continue;
        }

        for (; sum_index <= ecl_sum_get_last_report_step(ecl_sum);
             ++sum_index) {
            time_t sum_time = ecl_sum_get_report_time(ecl_sum, sum_index);
            if (sum_time == map_time)
                break;

            if (sum_time > map_time) {
                int day, month, year;
                util_set_date_values_utc(map_time, &day, &month, &year);
                util_abort("%s: The eclipse summary cases is missing data for "
                           "date:%4d-%02d-%02d - aborting\n",
                           __func__, year, month, day);
            }
        }

        if (sum_index > ecl_sum_get_last_report_step(ecl_sum)) {
            logger->error("Inconsistency in time_map - data will be ignored");
            break;
        }

        indices.push_back(sum_index);
    }

    return indices;
}

namespace {
using time_point =
    std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;

py::handle pybind_time(time_t time) {
    // Lazy initialise the PyDateTime import
    if (!PyDateTimeAPI) {
        PyDateTime_IMPORT;
    }

    std::tm tm;
    std::tm *tm_ptr = gmtime_r(&time, &tm);
    if (!tm_ptr) {
        throw py::cast_error("Unable to represent system_clock in local time");
    }
    return PyDateTime_FromDateAndTime(tm.tm_year + 1900, tm.tm_mon + 1,
                                      tm.tm_mday, tm.tm_hour, tm.tm_min,
                                      tm.tm_sec, 0);
}

py::handle pybind_getitem(TimeMap &self, size_t index) {
    time_t time = self.DEFAULT_TIME;
    if (index < self.size())
        time = self[index];
    return pybind_time(time);
}
} // namespace

ERT_CLIB_SUBMODULE("time_map", m) {
    using namespace py::literals;

    py::class_<TimeMap, std::shared_ptr<TimeMap>>(m, "TimeMap")
        .def(py::init<>())
        .def("__len__", [](TimeMap &self) { return self.size(); })
        .def("__getitem__", &pybind_getitem, "index"_a)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("read_text", &TimeMap::read_text, "path"_a)
        .def(
            "summary_update",
            [](TimeMap &self, Cwrap<ecl_sum_type> summary) {
                return self.summary_update(summary);
            },
            "summary"_a)
        .def(
            "attach_refcase",
            [](TimeMap &self, Cwrap<ecl_sum_type> summary) {
                return self.attach_refcase(summary);
            },
            "refcase"_a);
}
