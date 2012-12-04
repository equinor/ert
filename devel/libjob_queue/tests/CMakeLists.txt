add_executable( loadOK loadOK.c )
add_executable( loadFail loadFail.c )
add_executable( create_file create_file.c )
add_executable( job_workflow_test job_workflow_test.c )

target_link_libraries( job_workflow_test job_queue util )
target_link_libraries( create_file job_queue util )
target_link_libraries( loadOK job_queue util )
target_link_libraries( loadFail job_queue util )


add_test( job_workflow_test ${EXECUTABLE_OUTPUT_PATH}/job_workflow_test ${CMAKE_CURRENT_SOURCE_DIR}/data/external_job ${CMAKE_CURRENT_SOURCE_DIR}/data/internal_job)

add_test( loadOK ${EXECUTABLE_OUTPUT_PATH}/loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/internalOK)
add_test( loadOK ${EXECUTABLE_OUTPUT_PATH}/loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/externalOK)
add_test( loadOK ${EXECUTABLE_OUTPUT_PATH}/loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/internalOK ${CMAKE_CURRENT_SOURCE_DIR}/data/externalOK)

add_test( loadFail ${EXECUTABLE_OUTPUT_PATH}/loadFail ${CMAKE_CURRENT_SOURCE_DIR}/data/internalFail)
add_test( loadFail ${EXECUTABLE_OUTPUT_PATH}/loadFail ${CMAKE_CURRENT_SOURCE_DIR}/data/externalFail)
add_test( loadFail ${EXECUTABLE_OUTPUT_PATH}/loadFail ${CMAKE_CURRENT_SOURCE_DIR}/data/internalFail ${CMAKE_CURRENT_SOURCE_DIR}/data/externalFail)


