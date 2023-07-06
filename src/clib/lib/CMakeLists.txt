pybind11_add_module(
  _clib
  SHARED
  res_util/subst_list.cpp
  python/init.cpp
  python/logging.cpp
  job_queue/job_status.cpp
  job_queue/job_list.cpp
  job_queue/job_node.cpp
  job_queue/job_queue.cpp
  job_queue/job_queue_status.cpp
  job_queue/local_driver.cpp
  job_queue/lsf_driver.cpp
  job_queue/queue_driver.cpp
  job_queue/slurm_driver.cpp
  job_queue/torque_driver.cpp
  ${lsb}
  enkf/time_map.cpp
  enkf/config_keys.cpp
  enkf/enkf_obs.cpp
  enkf/enkf_types.cpp
  enkf/ensemble_config.cpp
  enkf/row_scaling.cpp)

# -----------------------------------------------------------------
# Target: Python C Extension 'ert._clib'
# -----------------------------------------------------------------

target_link_libraries(
  _clib PUBLIC std::filesystem cJSON::cJSON fmt::fmt Eigen3::Eigen
               cppitertools::cppitertools Threads::Threads)
target_include_directories(_clib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
                                        "${ECL_INCLUDE_DIRS}")

set_target_properties(_clib PROPERTIES CXX_VISIBILITY_PRESET "default")
install(TARGETS _clib LIBRARY DESTINATION src/ert)

# This extra deploy is only for ctest to be able to locate the script, this file
# is otherwise deployed through setup.py.
file(
  COPY "${CMAKE_CURRENT_SOURCE_DIR}/../../ert/job_queue/qstat_proxy.sh"
  DESTINATION "${CMAKE_BINARY_DIR}/old_tests/job_queue"
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)

# -----------------------------------------------------------------
# Target: 'libert.so' for use in tests
# -----------------------------------------------------------------

if(BUILD_TESTS)
  add_library(ert SHARED $<TARGET_OBJECTS:_clib>)
  target_link_libraries(ert _clib pybind11::embed ecl)
  fix_install_names(ert)
endif()
