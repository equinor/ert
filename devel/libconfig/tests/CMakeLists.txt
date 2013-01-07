add_executable( config_append_test config_append_test.c )
target_link_libraries( config_append_test config )

add_executable( config_node_test config_node_test.c )
target_link_libraries( config_node_test config )

add_executable( config_typeOK config_typeOK.c )
target_link_libraries( config_typeOK config )

add_executable( config_typeFail config_typeFail.c )
target_link_libraries( config_typeFail config )

add_executable( config_path_elm config_path_elm.c )
target_link_libraries( config_path_elm config )

add_executable( config_include_test config_include_test.c)
target_link_libraries( config_include_test config)

add_test( config_typeOK        ${EXECUTABLE_OUTPUT_PATH}/config_typeOK       ${CMAKE_CURRENT_SOURCE_DIR}/data/type_testOK )
add_test( config_typeFail      ${EXECUTABLE_OUTPUT_PATH}/config_typeFail     ${CMAKE_CURRENT_SOURCE_DIR}/data/type_testFail )
add_test( config_append_test   ${EXECUTABLE_OUTPUT_PATH}/config_append_test  ${CMAKE_CURRENT_SOURCE_DIR}/data/append_test )
add_test( config_node_test     ${EXECUTABLE_OUTPUT_PATH}/config_node_test    ${CMAKE_CURRENT_SOURCE_DIR}/data/append_test )
add_test( config_path_elm      ${EXECUTABLE_OUTPUT_PATH}/config_path_elm )
add_test( config_include_test  ${EXECUTABLE_OUTPUT_PATH}/config_include_test ${CMAKE_CURRENT_SOURCE_DIR}/data include_test )
