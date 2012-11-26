add_executable( append_test append_test.c )
target_link_libraries( append_test config )


add_test( append_test ${EXECUTABLE_OUTPUT_PATH}/append_test ${CMAKE_CURRENT_SOURCE_DIR}/data/append_test )
