add_executable( loadOK loadOK.c )
target_link_libraries( loadOK job_queue util )

add_executable( job_lsf_test job_lsf_test.c )
target_link_libraries( job_lsf_test job_queue util )

add_executable( job_lsf_submit_test job_lsf_submit_test.c )
target_link_libraries( job_lsf_submit_test job_queue util )

add_executable( job_program job_program.c )

add_test( loadOK ${EXECUTABLE_OUTPUT_PATH}/loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/internalOK)
add_test( job_lsf_test ${EXECUTABLE_OUTPUT_PATH}/job_lsf_test )
add_test( job_lsf_submit_test ${EXECUTABLE_OUTPUT_PATH}/job_lsf_submit_test ${EXECUTABLE_OUTPUT_PATH}/job_program )
