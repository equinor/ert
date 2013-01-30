add_executable( enkf_runpath_list enkf_runpath_list.c )
target_link_libraries( enkf_runpath_list enkf )

add_executable( enkf_site_config enkf_site_config.c )
target_link_libraries( enkf_site_config enkf )

add_executable( enkf_time_map enkf_time_map.c )
target_link_libraries( enkf_time_map enkf )

add_executable( enkf_ensemble_GEN_PARAM enkf_ensemble_GEN_PARAM.c )
target_link_libraries( enkf_ensemble_GEN_PARAM enkf )

add_executable( enkf_ensemble enkf_ensemble.c )
target_link_libraries( enkf_ensemble enkf )

add_executable( enkf_main enkf_main.c )
target_link_libraries( enkf_main enkf )
add_test( enkf_main  ${EXECUTABLE_OUTPUT_PATH}/enkf_main )

add_executable( enkf_model_config enkf_model_config.c )
target_link_libraries( enkf_model_config enkf )
add_test( enkf_model_config  ${EXECUTABLE_OUTPUT_PATH}/enkf_model_config )

add_executable( enkf_ecl_config enkf_ecl_config.c )
target_link_libraries( enkf_ecl_config enkf )
add_test( enkf_ecl_config1  ${EXECUTABLE_OUTPUT_PATH}/enkf_ecl_config )
add_test( enkf_ecl_config2  ${EXECUTABLE_OUTPUT_PATH}/enkf_ecl_config ${PROJECT_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE)
set_property( TEST enkf_ecl_config2 PROPERTY LABELS Statoil )


add_test( enkf_runpath_list  ${EXECUTABLE_OUTPUT_PATH}/enkf_runpath_list )
add_test( enkf_site_config  ${EXECUTABLE_OUTPUT_PATH}/enkf_site_config /project/res/etc/ERT/site-config)
add_test( enkf_time_map1  ${EXECUTABLE_OUTPUT_PATH}/enkf_time_map )
add_test( enkf_time_map2  ${EXECUTABLE_OUTPUT_PATH}/enkf_time_map ${PROJECT_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE.UNRST )
add_test( enkf_ensemble_GEN_PARAM  ${EXECUTABLE_OUTPUT_PATH}/enkf_ensemble_GEN_PARAM ${CMAKE_CURRENT_SOURCE_DIR}/data/ensemble/GEN_PARAM )
add_test( enkf_ensemble  ${EXECUTABLE_OUTPUT_PATH}/enkf_ensemble )


set_property( TEST enkf_time_map2   PROPERTY LABELS Statoil )

