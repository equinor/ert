add_executable(ies_enkf_config ${TESTS_EXCLUDE_FROM_ALL}
                               ies/test_ies_enkf_config.cpp)
target_link_libraries(ies_enkf_config ert)
add_test(NAME ies_enkf_config COMMAND ies_enkf_config)

add_executable(ies_module ${TESTS_EXCLUDE_FROM_ALL}
                          ies/test_ies_enkf_module.cpp)
target_link_libraries(ies_module ert)
add_test(NAME ies_module COMMAND ies_module
                                 "${TEST_DATA_DIR}/analysis/ies/poly")
