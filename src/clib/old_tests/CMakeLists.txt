if(NOT BUILD_TESTS)
  return()
endif()

# -----------------------------------------------------------------
# Additional files setup
# -----------------------------------------------------------------

file(
  COPY "${CMAKE_SOURCE_DIR}/../_ert_job_runner/job_dispatch.py"
  DESTINATION "${CMAKE_BINARY_DIR}/bin"
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)

# -----------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------

set(SITE_CONFIG_PATH "${CMAKE_SOURCE_DIR}/../ert/shared/share/ert/site-config")

set(TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/old_tests/data")
set(SHARE_DIR "${CMAKE_SOURCE_DIR}/../ert/shared/share/ert")

# -----------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------

add_subdirectory(enkf)
add_subdirectory(job_queue)
add_subdirectory(res_util)
