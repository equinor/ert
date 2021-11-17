if(NOT BUILD_TESTS)
  return()
endif()

find_package(Eigen3 REQUIRED)

find_package(Catch2 REQUIRED)
include(Catch)

add_executable(ert_test_suite analysis/modules/test_ies_enkf_main.cpp test_matrix.cpp)
target_link_libraries(ert_test_suite res Catch2::Catch2WithMain Eigen3::Eigen)

catch_discover_tests(ert_test_suite)
