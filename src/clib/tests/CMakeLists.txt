if(NOT BUILD_TESTS)
  return()
endif()

find_package(Catch2 REQUIRED)
include(Catch)

add_executable(
  ert_test_suite ${TESTS_EXCLUDE_FROM_ALL} tmpdir.cpp res_util/test_string.cpp
                 job_queue/test_lsf_driver.cpp)

target_compile_definitions(
  ert_test_suite
  PRIVATE
    "LOCAL_TEST_DATA_DIR=::std::filesystem::path(\"${CMAKE_SOURCE_DIR}/../../test-data\")"
)

target_link_libraries(ert_test_suite ert Catch2::Catch2WithMain fmt::fmt)
fix_install_names(ert_test_suite)

catch_discover_tests(ert_test_suite)
