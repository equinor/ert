#include <ert/job_queue/job_status.hpp>
#include <ert/job_queue/torque_driver.hpp>
#include <ert/python.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

ERT_CLIB_SUBMODULE("_torque_driver", m) {
    py::enum_<job_status_type>(m, "JobStatusType", py::arithmetic())
        .export_values();
    m.def("torque_job_create_submit_script",
          [](std::string script_filename, std::string submit_cmd,
             std::vector<std::string> job_argv) {
              std::vector<const char *> strings;
              for (int i = 0; i < job_argv.size(); ++i)
                  strings.push_back(job_argv[i].c_str());
              torque_job_create_submit_script(script_filename.c_str(),
                                              submit_cmd.c_str(),
                                              strings.size(), strings.data());
          });

    m.def("torque_driver_parse_status",
          [](std::string qstat_file, std::string jobnr_char) {
              auto result = torque_driver_parse_status(qstat_file.c_str(),
                                                       jobnr_char.c_str());
              return result;
          });
}
