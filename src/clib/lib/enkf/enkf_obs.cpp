#include <map>
#include <utility>

#include <ert/python.hpp>
#include <ert/res_util/string.hpp>
#include <resdata/rd_sum.h>

ERT_CLIB_SUBMODULE("enkf_obs", m) {
    using namespace py::literals;
    m.def("read_from_refcase",
          [](Cwrap<rd_sum_type> refcase, std::string local_key) {
              int num_steps = rd_sum_get_last_report_step(refcase);
              std::vector<bool> valid(num_steps + 1);
              std::vector<double> value(num_steps + 1);
              for (int tstep = 0; tstep <= num_steps; tstep++) {
                  if (rd_sum_has_report_step(refcase, tstep)) {
                      int time_index = rd_sum_iget_report_end(refcase, tstep);
                      value[tstep] = rd_sum_get_general_var(refcase, time_index,
                                                            local_key.c_str());
                      valid[tstep] = true;
                  } else {
                      valid[tstep] = false;
                  }
              }

              return std::make_pair(valid, value);
          });
}
