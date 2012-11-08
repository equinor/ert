add_executable( loadOK loadOK.c )
target_link_libraries( loadOK job_queue util )


add_test( loadOK ${EXECUTABLE_OUTPUT_PATH}/loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/internalOK)
