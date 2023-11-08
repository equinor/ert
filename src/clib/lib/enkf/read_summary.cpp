#include <algorithm>
#include <ert/python.hpp>
#include <fnmatch.h>
#include <pybind11/pybind11.h>
#include <resdata/rd_smspec.hpp>
#include <resdata/rd_sum.hpp>
#include <stdexcept>
#include <string>
#include <time.h>
#include <tuple>
#include <vector>

#include <datetime.h> // must be included after pybind11.h

static bool matches(const std::vector<std::string> &patterns, const char *key) {
    return std::any_of(patterns.cbegin(), patterns.cend(),
                       [key](const std::string &pattern) {
                           return fnmatch(pattern.c_str(), key, 0) == 0;
                       });
}

ERT_CLIB_SUBMODULE("_read_summary", m) {
    m.def("read_dates", [](Cwrap<rd_sum_type> summary) {
        if (!PyDateTimeAPI)
            PyDateTime_IMPORT;

        time_t_vector_type *tvec = rd_sum_alloc_time_vector(summary, true);
        int size = time_t_vector_size(tvec);
        pybind11::list result(size);
        auto t = tm{};
        for (int i = 0; i < size; i++) {
            auto timestamp = time_t_vector_iget(tvec, i);
            auto success = ::gmtime_r(&timestamp, &t);
            if (success == nullptr)
                throw std::runtime_error("Unable to parse unix timestamp: " +
                                         std::to_string(timestamp));

            if (!PyDateTimeAPI) // this is here to silence the linters. it will always be set.
                throw std::runtime_error("Python DateTime API not loaded");

            auto py_time = PyDateTime_FromDateAndTime(
                t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min,
                t.tm_sec, 0);
            if (py_time == nullptr)
                throw std::runtime_error("Unable to create DateTime object "
                                         "from unix timestamp: " +
                                         std::to_string(timestamp));

            result[i] = pybind11::reinterpret_steal<pybind11::object>(py_time);
        }

        time_t_vector_free(tvec);
        return result;
    });
    m.def("read_summary", [](Cwrap<rd_sum_type> summary,
                             std::vector<std::string> keys) {
        const rd_smspec_type *smspec = rd_sum_get_smspec(summary);
        std::vector<std::pair<std::string, std::vector<double>>>
            summary_vectors{};
        std::vector<std::string> seen_keys{};
        for (int i = 0; i < rd_smspec_num_nodes(smspec); i++) {
            const rd::smspec_node &smspec_node =
                rd_smspec_iget_node_w_node_index(smspec, i);
            const char *key = smspec_node.get_gen_key1();
            if ((matches(keys, key)) &&
                !(std::find(seen_keys.begin(), seen_keys.end(), key) !=
                  seen_keys.end())) {
                seen_keys.push_back(key);
                int start = rd_sum_get_first_report_step(summary);
                int end = rd_sum_get_last_report_step(summary);
                std::vector<double> data{};
                int key_index =
                    rd_sum_get_general_var_params_index(summary, key);
                for (int tstep = start; tstep <= end; tstep++) {
                    if (rd_sum_has_report_step(summary, tstep)) {
                        int time_index = rd_sum_iget_report_end(summary, tstep);
                        data.push_back(
                            rd_sum_iget(summary, time_index, key_index));
                    }
                }
                summary_vectors.emplace_back(key, data);
            }
        }
        return summary_vectors;
    });
}
