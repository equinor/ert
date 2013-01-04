add_executable( job_loadOK job_loadOK.c )
add_executable( job_loadFail job_loadFail.c )
add_executable( create_file create_file.c )
add_executable( job_workflow_test job_workflow_test.c )

target_link_libraries( job_workflow_test job_queue util )
target_link_libraries( create_file job_queue util )
target_link_libraries( job_loadOK job_queue util )
target_link_libraries( job_loadFail job_queue util )


add_test( job_workflow_test ${EXECUTABLE_OUTPUT_PATH}/job_workflow_test ${CMAKE_CURRENT_SOURCE_DIR}/data/external_job ${CMAKE_CURRENT_SOURCE_DIR}/data/internal_job)

add_test( job_loadOK1 ${EXECUTABLE_OUTPUT_PATH}/job_loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/internalOK)
add_test( job_loadOK2 ${EXECUTABLE_OUTPUT_PATH}/job_loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/externalOK)
add_test( job_loadOK3 ${EXECUTABLE_OUTPUT_PATH}/job_loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/internalOK ${CMAKE_CURRENT_SOURCE_DIR}/data/externalOK)

add_test( job_loadFail1 ${EXECUTABLE_OUTPUT_PATH}/job_loadFail ${CMAKE_CURRENT_SOURCE_DIR}/data/internalFail)
add_test( job_loadFail2 ${EXECUTABLE_OUTPUT_PATH}/job_loadFail ${CMAKE_CURRENT_SOURCE_DIR}/data/externalFail)
add_test( job_loadFail3 ${EXECUTABLE_OUTPUT_PATH}/job_loadFail ${CMAKE_CURRENT_SOURCE_DIR}/data/internalFail ${CMAKE_CURRENT_SOURCE_DIR}/data/externalFail)

add_executable( job_lsf_test job_lsf_test.c )
target_link_libraries( job_lsf_test job_queue util )

add_executable( job_lsf_submit_test job_lsf_submit_test.c )
target_link_libraries( job_lsf_submit_test job_queue util )

add_executable( job_program job_program.c )

add_test( job_lsf_test ${EXECUTABLE_OUTPUT_PATH}/job_lsf_test )
add_test( job_lsf_submit_test ${EXECUTABLE_OUTPUT_PATH}/job_lsf_submit_test ${EXECUTABLE_OUTPUT_PATH}/job_program )
