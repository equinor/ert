# -----------------------------------------------------------------
find_program(SBATCH "sbatch")

set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

file(COPY "${TEST_DATA_DIR}/job_queue/qsub_emulators/"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

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
  ert_test_suite
  ${TESTS_EXCLUDE_FROM_ALL}
  job_queue/test_job_list.cpp
  job_queue/test_job_lsf.cpp
  job_queue/test_job_lsf_parse_bsub_stdout.cpp
  job_queue/test_job_mock_slurm.cpp
  job_queue/test_job_queue_driver.cpp
  job_queue/test_job_slurm_driver.cpp
  $<$<BOOL:${SBATCH}>:job_queue/test_job_slurm_submit.cpp> # if found add file
  $<$<BOOL:${SBATCH}>:job_queue/test_job_slurm_runtest.cpp> # if found add file
  job_queue/test_job_torque.cpp
  job_queue/test_job_torque_submit.cpp
  job_queue/test_lsf_driver.cpp
  enkf/test_row_scaling.cpp
  res_util/test_string.cpp
  tmpdir.cpp)

set_source_files_properties(
  test_job_torque_submit.cpp
  PROPERTIES ENVIRONMENT "setenv PATH ${CMAKE_CURRENT_BINARY_DIR}:$ENV{PATH}")

target_compile_definitions(
  ert_test_suite
  PRIVATE
    "LOCAL_TEST_DATA_DIR=::std::filesystem::path(\"${CMAKE_SOURCE_DIR}/../../test-data\")"
)

target_link_libraries(ert_test_suite ert Catch2::Catch2WithMain fmt::fmt)

catch_discover_tests(ert_test_suite)
