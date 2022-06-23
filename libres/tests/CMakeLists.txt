if(NOT BUILD_TESTS)
  return()
endif()

find_package(Catch2 REQUIRED)
include(Catch)

add_executable(
  ert_test_suite
  tmpdir.cpp
  analysis/ies/test_ies_enkf_main.cpp
  analysis/test_enkf_linalg.cpp
  analysis/test_save_parameters.cpp
  analysis/test_copy_parameters.cpp
  enkf/enkf_obs_paths_detailed.cpp
  enkf/test_enkf_fs.cpp
  enkf/test_analysis_config.cpp
  enkf/test_meas_data.cpp
  enkf/test_obs_data.cpp
  enkf/test_deprecated_umask.cpp
  res_util/test_memory.cpp
  res_util/test_string.cpp
  res_util/test_metric.cpp
  analysis/test_update.cpp
  job_queue/test_lsf_driver.cpp
  job_queue/test_rsh_driver.cpp
  job_queue/test_ext_job_executable.cpp)

target_link_libraries(ert_test_suite res Catch2::Catch2WithMain fmt::fmt)

catch_discover_tests(ert_test_suite)
