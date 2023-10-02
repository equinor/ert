#include <chrono>
#include <ert/ecl/ecl_smspec.hpp>
#include <ert/ecl/ecl_sum.hpp>
#include <ert/python.hpp>
#include <fnmatch.h>
#include <string>
#include <tuple>
#include <vector>

static bool matches(std::vector<std::string> patterns, std::string key) {
    bool has_key = false;
    for (auto pattern : patterns) {
        if (fnmatch(pattern.c_str(), key.c_str(), 0) == 0) {
            has_key = true;
            break;
        }
    }
    return has_key;
}
ERT_CLIB_SUBMODULE("_read_summary", m) {
    m.def("read_dates", [](Cwrap<ecl_sum_type> summary) {
        time_t_vector_type *tvec = ecl_sum_alloc_time_vector(summary, true);
        int size = time_t_vector_size(tvec);
        std::vector<std::chrono::system_clock::time_point> result(size);
        for (int i = 0; i < size; i++) {
            result[i] = std::chrono::system_clock::from_time_t(
                time_t_vector_iget(tvec, i));
        }
        return result;
    });
    m.def("read_summary",
          [](Cwrap<ecl_sum_type> summary, std::vector<std::string> keys) {
              const int step2 = ecl_sum_get_last_report_step(summary);
              const ecl_smspec_type *smspec = ecl_sum_get_smspec(summary);
              std::vector<std::pair<std::string, std::vector<double>>>
                  summary_vectors{};
              std::vector<std::string> seen_keys{};
              for (int i = 0; i < ecl_smspec_num_nodes(smspec); i++) {
                  const ecl::smspec_node &smspec_node =
                      ecl_smspec_iget_node_w_node_index(smspec, i);
                  const char *key = smspec_node.get_gen_key1();
                  if ((matches(keys, key)) &&
                      !(std::find(seen_keys.begin(), seen_keys.end(), key) !=
                        seen_keys.end())) {
                      seen_keys.push_back(key);
                      int start = ecl_sum_get_first_report_step(summary);
                      int end = ecl_sum_get_last_report_step(summary);
                      std::vector<double> data{};
                      int key_index =
                          ecl_sum_get_general_var_params_index(summary, key);
                      for (int tstep = start; tstep <= end; tstep++) {
                          if (ecl_sum_has_report_step(summary, tstep)) {
                              int time_index =
                                  ecl_sum_iget_report_end(summary, tstep);
                              data.push_back(
                                  ecl_sum_iget(summary, time_index, key_index));
                          }
                      }
                      summary_vectors.emplace_back(key, data);
                  }
              }
              return summary_vectors;
          });
}
