unset(VAR_RML CACHE)
unset(VAR_RMLI CACHE)
ert_module_name( VAR_RML  rml_enkf  ${LIBRARY_OUTPUT_PATH} )
ert_module_name( VAR_RMLI rmli_enkf ${LIBRARY_OUTPUT_PATH} )

add_executable(analysis_test_external_module analysis_test_external_module.c )
target_link_libraries( analysis_test_external_module analysis util test_util )

add_test( analysis_module_rml  ${EXECUTABLE_OUTPUT_PATH}/analysis_test_external_module "RML_ENKF"  ${VAR_RML} 41 ITER LAMBDA0)
add_test( analysis_module_rmli ${EXECUTABLE_OUTPUT_PATH}/analysis_test_external_module "RMLI_ENKF" ${VAR_RMLI} 41 ITER)


