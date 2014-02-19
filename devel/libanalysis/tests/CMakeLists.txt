ert_module_name( VAR_RMLI rmli_enkf ${LIBRARY_OUTPUT_PATH} )

add_executable(analysis_test_external_module analysis_test_external_module.c )
target_link_libraries( analysis_test_external_module analysis util test_util )

add_test( analysis_module_rmli ${EXECUTABLE_OUTPUT_PATH}/analysis_test_external_module "RMLI_ENKF" ${VAR_RMLI} 41 ITER USE_PRIOR)


