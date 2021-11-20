if(NOT BUILD_TESTS)
  return()
endif()

find_package(Catch2 REQUIRED)

add_executable(ert_test_suite analysis/modules/test_ies_enkf_main.cpp
analysis/test_enkf_linalg.cpp
res_util/test_matrix.cpp)
target_link_libraries(ert_test_suite res Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(ert_test_suite)
