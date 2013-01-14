add_executable( enkf_runpath_list enkf_runpath_list.c )
target_link_libraries( enkf_runpath_list enkf )

add_test( enkf_runpath_list  ${EXECUTABLE_OUTPUT_PATH}/enkf_runpath_list )
