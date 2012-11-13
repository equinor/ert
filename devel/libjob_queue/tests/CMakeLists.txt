add_executable( loadOK loadOK.c )
add_executable( loadFail loadFail.c )
target_link_libraries( loadOK job_queue util )
target_link_libraries( loadFail job_queue util )


add_test( loadOK ${EXECUTABLE_OUTPUT_PATH}/loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/internalOK)
add_test( loadOK ${EXECUTABLE_OUTPUT_PATH}/loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/externalOK)
add_test( loadOK ${EXECUTABLE_OUTPUT_PATH}/loadOK ${CMAKE_CURRENT_SOURCE_DIR}/data/internalOK ${CMAKE_CURRENT_SOURCE_DIR}/data/externalOK)

add_test( loadFail ${EXECUTABLE_OUTPUT_PATH}/loadFail ${CMAKE_CURRENT_SOURCE_DIR}/data/internalFail)
add_test( loadFail ${EXECUTABLE_OUTPUT_PATH}/loadFail ${CMAKE_CURRENT_SOURCE_DIR}/data/externalFail)
add_test( loadFail ${EXECUTABLE_OUTPUT_PATH}/loadFail ${CMAKE_CURRENT_SOURCE_DIR}/data/internalFail ${CMAKE_CURRENT_SOURCE_DIR}/data/externalFail)


