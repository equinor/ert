if(NOT BUILD_TESTS)
  return()
endif()

find_package(Catch2 REQUIRED)
include(Catch)

find_package(stduuid REQUIRED)
# include(stduuid)

add_executable(ert_test_suite analysis/modules/test_ies_enkf_main.cpp
                              analysis/test_enkf_linalg.cpp
                              analysis/test_fwd_step_log.cpp
                              res_util/test_matrix.cpp)
target_link_libraries(ert_test_suite res Catch2::Catch2WithMain stduuid)

catch_discover_tests(ert_test_suite)
