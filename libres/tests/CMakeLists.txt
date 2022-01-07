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
  enkf/enkf_obs_paths_detailed.cpp
  enkf/test_cases_config.cpp
  enkf/test_enkf_fs.cpp
  enkf/test_analysis_config.cpp
  enkf/test_obs_data.cpp
  res_util/test_matrix.cpp
  res_util/test_memory.cpp
  analysis/test_std_enkf.cpp
  analysis/test_compare_initX.cpp
  analysis/test_update.cpp)

target_link_libraries(ert_test_suite res Catch2::Catch2WithMain fmt::fmt)

catch_discover_tests(ert_test_suite)
