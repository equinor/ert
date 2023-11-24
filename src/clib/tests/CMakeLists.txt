# -----------------------------------------------------------------
find_program(SBATCH "sbatch")

set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

file(
  COPY "${CMAKE_SOURCE_DIR}/../_ert_job_runner/job_dispatch.py"
  DESTINATION "${CMAKE_BINARY_DIR}/bin"
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
# -----------------------------------------------------------------

if(NOT BUILD_TESTS)
  return()
endif()

find_package(Catch2 REQUIRED)
include(Catch)

add_executable(
  ert_test_suite ${TESTS_EXCLUDE_FROM_ALL} enkf/test_row_scaling.cpp
                 res_util/test_string.cpp tmpdir.cpp)

target_compile_definitions(
  ert_test_suite
  PRIVATE
    "LOCAL_TEST_DATA_DIR=::std::filesystem::path(\"${CMAKE_SOURCE_DIR}/../../test-data\")"
)

target_link_libraries(ert_test_suite ert Catch2::Catch2WithMain fmt::fmt)
fix_install_names(ert_test_suite)

catch_discover_tests(ert_test_suite)
