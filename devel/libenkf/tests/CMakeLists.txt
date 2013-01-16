add_executable( enkf_runpath_list enkf_runpath_list.c )
target_link_libraries( enkf_runpath_list enkf )

add_executable( enkf_time_map enkf_time_map.c )
target_link_libraries( enkf_time_map enkf )

add_test( enkf_runpath_list  ${EXECUTABLE_OUTPUT_PATH}/enkf_runpath_list )
add_test( enkf_time_map1  ${EXECUTABLE_OUTPUT_PATH}/enkf_time_map )
add_test( enkf_time_map2  ${EXECUTABLE_OUTPUT_PATH}/enkf_time_map ${PROJECT_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE.UNRST )

set_property( TEST enkf_time_map2 PROPERTY LABELS Statoil )
