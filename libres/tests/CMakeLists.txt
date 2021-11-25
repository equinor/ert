if(NOT BUILD_TESTS)
  return()
endif()

find_package(Catch2 REQUIRED)
find_package(fmt REQUIRED)
include(Catch)

add_executable(ert_test_suite tmpdir.cpp
                              analysis/modules/test_ies_enkf_main.cpp
                              analysis/test_enkf_linalg.cpp
                              res_util/test_matrix.cpp
							  enkf/enkf_obs_paths_detailed.cpp
							  )
target_link_libraries(ert_test_suite res Catch2::Catch2WithMain fmt::fmt)

catch_discover_tests(ert_test_suite)
