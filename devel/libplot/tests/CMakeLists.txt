add_executable( plot_text plot_text.c )
target_link_libraries( plot_text plot )
add_test( plot_text ${EXECUTABLE_OUTPUT_PATH}/plot_text )
