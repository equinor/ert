if (BUILD_APPLICATIONS)
	unset(VAR_RML CACHE)
	unset(VAR_RMLI CACHE)
	find_library(VAR_RML NAMES rml_enkf.dylib rml_enkf.so PATHS ${LIBRARY_OUTPUT_PATH} )
	add_test( analysis_module_test_RML ${EXECUTABLE_OUTPUT_PATH}/ert_module_test ${VAR_RML})
	find_library(VAR_RMLI NAMES rmli_enkf.so rmli_enkf.dylib PATHS ${LIBRARY_OUTPUT_PATH} )
	add_test( analysis_module_test_RMLI ${EXECUTABLE_OUTPUT_PATH}/ert_module_test ${VAR_RMLI} )
endif()

add_executable(analysis_test_external_module analysis_test_external_module.c )
target_link_libraries( analysis_test_external_module analysis util test_util )

add_test( analysis_module_rml  ${EXECUTABLE_OUTPUT_PATH}/analysis_test_external_module "RML_ENKF" ${VAR_RML} 41 ITER LAMBDA0)
add_test( analysis_module_rmli ${EXECUTABLE_OUTPUT_PATH}/analysis_test_external_module "RMLI_ENKF" ${VAR_RMLI} 41 ITER)


