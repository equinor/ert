if(NOT BUILD_TESTS)
  return()
endif()

# -----------------------------------------------------------------
# Additional files setup
# -----------------------------------------------------------------

file(
  COPY "${CMAKE_SOURCE_DIR}/../ert/job_runner/job_dispatch.py"
  DESTINATION "${CMAKE_BINARY_DIR}/bin"
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)

# -----------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------

set(SITE_CONFIG_PATH "${CMAKE_SOURCE_DIR}/../ert/shared/share/ert/site-config")

set(EQUINOR_TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/../../test-data/Equinor")
set(TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/old_tests/data")
set(LOCAL_TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/../../test-data/local")
set(SHARE_DIR "${CMAKE_SOURCE_DIR}/../ert/shared/share/ert")

# -----------------------------------------------------------------
# -DEQUINOR_TESTDATA_ROOT support
# -----------------------------------------------------------------

set(EQUINOR_TESTDATA_ROOT
    ""
    CACHE PATH "Root to Equinor internal testdata")
if(EXISTS "${EQUINOR_TESTDATA_ROOT}")
  if(EXISTS "${EQUINOR_TEST_DATA_DIR}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E remove
                            "${EQUINOR_TEST_DATA_DIR}")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                          "${EQUINOR_TESTDATA_ROOT}" "${EQUINOR_TEST_DATA_DIR}")
  message(
    STATUS
      "Linking testdata: ${EQUINOR_TEST_DATA_DIR} -> ${EQUINOR_TESTDATA_ROOT}")
endif()

# -----------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------

add_subdirectory(analysis)
add_subdirectory(config)
add_subdirectory(enkf)
add_subdirectory(job_queue)
add_subdirectory(res_util)
add_subdirectory(rms)
add_subdirectory(sched)
